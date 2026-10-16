"""The whole pipeline, end to end, with no model and no network.

Every other test covers one seam. This one covers the joins between them: corpus files
-> extracted text -> overlapping chunks -> vectors -> FAISS + BM25 -> RRF -> a grounded
prompt -> an answer -> citations -> five scored metrics -> a rendered report.

It runs offline because the embedder is :class:`mmu.embed.hashing.HashingEmbedder` and
the generator is a scripted fake. That is the point: the integration bugs this catches —
a chunk_id that does not round-trip, a manifest that refuses its own index, a marker
that resolves to the wrong document, a metric that receives the wrong shape — have
nothing to do with model quality, and waiting on a 2.3GB download to find them would
mean nobody runs this.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict

import pytest

from mmu.config import Settings
from mmu.embed.hashing import HashingEmbedder
from mmu.eval import report as report_mod
from mmu.eval.judge import LocalJudge
from mmu.eval.questions import load_questions, validate_questions
from mmu.eval.runner import run_eval
from mmu.generate.prompt import build_messages, parse_markers, resolve_citations
from mmu.retrieve.build import build_index, load_index
from mmu.retrieve.hybrid import HybridRetriever
from tests.conftest import FakeGenerator

QUESTIONS = """\
{"id":"xdr-001","question":"How fast must we notify and who owns it?",\
"answer_type":"cross_document","required_doc_ids":["gdpr","company-policy"],\
"relevant_doc_ids":["gdpr","company-policy"],\
"must_contain":{"gdpr":["72 hours"],"company-policy":["Data Protection Officer"]},\
"expected_synthesis_keys":["72 hours","Data Protection Officer"],"difficulty":"hard"}
{"id":"single-001","question":"What is the statutory notification deadline?",\
"answer_type":"single_document","required_doc_ids":["gdpr"],"relevant_doc_ids":["gdpr"],\
"must_contain":{"gdpr":["72 hours"]},"expected_synthesis_keys":["72 hours"]}
"""


@pytest.fixture
def built(tmp_path, mini_corpus):
    embedder = HashingEmbedder(dim=64)
    index_dir = tmp_path / ".index"
    store, dense, sparse, warnings = build_index(
        mini_corpus, index_dir, embedder, max_tokens=120, min_tokens=20, overlap_tokens=16
    )
    return embedder, index_dir, store, dense, sparse, warnings


def test_index_builds_and_round_trips_through_disk(built) -> None:
    embedder, index_dir, store, dense, sparse, warnings = built
    assert warnings == []
    assert len(store) > 0
    assert set(store.doc_ids()) == {"gdpr", "national-law", "company-policy"}
    assert dense.size == len(store)

    # The chunk/vector row correspondence is what everything downstream assumes.
    reloaded_store, reloaded_dense, reloaded_sparse = load_index(index_dir, embedder)
    assert len(reloaded_store) == len(store)
    assert reloaded_dense.size == dense.size
    assert [c.chunk_id for c in reloaded_store.chunks] == [c.chunk_id for c in store.chunks]


def test_chunk_text_matches_its_own_offsets(built) -> None:
    """The invariant that makes a citation able to point at real source text."""
    _, _, store, _, _, _ = built
    for chunk in store.chunks:
        assert chunk.char_end > chunk.char_start
        assert len(chunk.text) == chunk.char_end - chunk.char_start


def test_hybrid_retrieval_finds_the_cross_document_evidence(built) -> None:
    embedder, _, store, dense, sparse, _ = built
    retriever = HybridRetriever(store, dense, sparse, embedder)
    result = retriever.search("breach notification deadline supervisory authority", k=8)

    assert result.chunks
    # The question is unanswerable from one document; retrieval must surface several.
    assert len({c.doc_id for c in result.chunks}) >= 2
    assert all(c.dense_rank or c.sparse_rank for c in result.chunks)


def test_prompt_markers_resolve_back_to_the_right_documents(built) -> None:
    embedder, _, store, dense, sparse, _ = built
    retriever = HybridRetriever(store, dense, sparse, embedder)
    chunks = retriever.search("notification deadline", k=4).chunks

    messages = build_messages("When must we notify?", chunks)
    assert any("[S1]" in m.content for m in messages)

    answer = "Notify within 72 hours [S1], and escalate internally [S2]."
    assert parse_markers(answer, len(chunks)) == [1, 2]
    citations = resolve_citations(answer, chunks)
    assert [c.marker for c in citations] == [1, 2]
    assert citations[0].doc_id == chunks[0].doc_id
    assert citations[1].doc_id == chunks[1].doc_id
    # An out-of-range marker is dropped, not crashed on.
    assert parse_markers("see [S99]", len(chunks)) == []


def test_questions_validate_against_the_built_index(tmp_path, built) -> None:
    _, _, store, _, _, _ = built
    path = tmp_path / "questions.jsonl"
    path.write_text(QUESTIONS, encoding="utf-8")

    questions = load_questions(path)
    report = validate_questions(
        questions,
        known_doc_ids=store.doc_ids(),
        doc_texts={d: store.text_of_doc(d) for d in store.doc_ids()},
    )
    assert report.ok, report.errors


def test_validate_rejects_an_unknown_doc_id(tmp_path, built) -> None:
    """The hard gate. Without it a typo reads as a model failure for an hour."""
    _, _, store, _, _, _ = built
    path = tmp_path / "questions.jsonl"
    path.write_text(QUESTIONS.replace('"gdpr"', '"gdrp"'), encoding="utf-8")

    report = validate_questions(
        load_questions(path),
        known_doc_ids=store.doc_ids(),
        doc_texts={d: store.text_of_doc(d) for d in store.doc_ids()},
    )
    assert not report.ok
    assert any("gdrp" in e for e in report.errors)


async def test_full_eval_run_produces_all_five_metrics(tmp_path, built) -> None:
    embedder, _, store, dense, sparse, _ = built
    retriever = HybridRetriever(store, dense, sparse, embedder)
    path = tmp_path / "questions.jsonl"
    path.write_text(QUESTIONS, encoding="utf-8")

    generator = FakeGenerator(
        reply=(
            "The controller must notify the supervisory authority within 72 hours [S1]. "
            "Internally, escalate to the Data Protection Officer [S2]."
        )
    )
    settings = Settings(data_dir=tmp_path)
    result = await run_eval(
        load_questions(path), retriever, generator, LocalJudge(), settings,
        k=6, cache_path=tmp_path / "judgments.jsonl",
    )

    assert len(result.runs) == 2
    # All five metrics are present and in range.
    assert 0.0 <= result.af <= 1.0
    assert 0.0 <= result.cp <= 1.0
    assert 0.0 <= result.xdr_mean <= 1.0
    assert 0.0 <= result.cur_cited <= 1.0
    assert result.corpus_lpqu > 0.0

    # XDR covers only the cross-document question, and reports its n.
    assert result.xdr_n == 1
    xdr_run = next(r for r in result.runs if r.question_id == "xdr-001")
    assert xdr_run.xdr is not None
    # The four factors are carried separately, never collapsed to the product.
    assert xdr_run.xdr.xdr == pytest.approx(
        xdr_run.xdr.coverage * xdr_run.xdr.cite * xdr_run.xdr.synthesis
    )
    # A single-document question is excluded rather than scored zero.
    assert next(r for r in result.runs if r.question_id == "single-001").xdr is None


async def test_report_renders_with_its_caveats(tmp_path, built) -> None:
    embedder, _, store, dense, sparse, _ = built
    retriever = HybridRetriever(store, dense, sparse, embedder)
    path = tmp_path / "questions.jsonl"
    path.write_text(QUESTIONS, encoding="utf-8")

    settings = Settings(data_dir=tmp_path)
    result = await run_eval(
        load_questions(path), retriever, FakeGenerator(), LocalJudge(), settings, k=4
    )
    text = report_mod.render(result, verbose=True)

    # The three honesty rules the report exists to enforce.
    assert "Cov (retrieval)" in text and "Cite (generation)" in text and "Syn" in text
    assert "negation-blind" in text, "a local-judge run must carry its caveat"
    # The span-derived CUR columns are structurally zero under LocalJudge; the report
    # must say so rather than letting a 0.000 read as "this model fakes its citations".
    assert "local judge emits no spans" in text
    assert "NOT " in text and "structurally zero" in text
    assert "noise band" in text or "not results" in text
    assert "SUM(t)/SUM(Q)" in text

    # And it round-trips to disk for `mmu eval compare`.
    out = tmp_path / "run.json"
    report_mod.save(result, out)
    assert report_mod.load(out)["xdr_n"] == result.xdr_n


async def test_unanswerable_question_does_not_cap_corpus_precision(tmp_path, built) -> None:
    """An unanswerable question has no relevant set, so Context Precision is undefined.

    Scoring it 0.0 and folding that into the mean would cap corpus CP below 1.0 purely
    because such a question exists — a perfect retriever would be reported as imperfect,
    and CP would stop being comparable across question sets with different proportions
    of unanswerable questions. It is excluded and the count is reported instead.
    """
    embedder, _, store, dense, sparse, _ = built
    retriever = HybridRetriever(store, dense, sparse, embedder)
    path = tmp_path / "questions.jsonl"
    path.write_text(
        QUESTIONS
        + '{"id":"unanswerable-001","question":"What is the maximum fine?",'
          '"answer_type":"unanswerable","required_doc_ids":[],"relevant_doc_ids":[],'
          '"must_contain":{},"expected_synthesis_keys":[]}\n',
        encoding="utf-8",
    )

    settings = Settings(data_dir=tmp_path)
    result = await run_eval(
        load_questions(path), retriever, FakeGenerator(), LocalJudge(), settings, k=6
    )

    assert len(result.runs) == 3
    unanswerable = next(r for r in result.runs if r.question_id == "unanswerable-001")
    assert unanswerable.cp is None, "CP is undefined without a relevant set, not zero"
    assert unanswerable.xdr is None

    # CP averages only the two questions that have a relevant set.
    assert result.cp_n == 2
    scored = [r.cp for r in result.runs if r.cp is not None]
    assert result.cp == pytest.approx(sum(scored) / len(scored))

    # And its quality is the AF-only renormalization, still on a 0-1 scale.
    assert unanswerable.quality == pytest.approx(unanswerable.af)
    assert 0.0 <= unanswerable.quality <= 1.0

    rendered = report_mod.render(result, verbose=True)
    assert "n=2 question(s) with a relevant set" in rendered
    # A dimension a question does not have reads as "not applicable", never as a nan
    # that looks like a computation went wrong. Matched as a standalone token, because
    # "unanswerable" contains the substring "nan".
    assert not re.search(r"(?<![A-Za-z])nan(?![A-Za-z])", rendered), rendered
    assert "--" in rendered


@pytest.mark.anyio
async def test_a_correct_refusal_citing_nothing_scores_full_faithfulness(tmp_path, built) -> None:
    """questions.jsonl's unanswerable-001 states the contract: "A correct refusal must
    score 1.0 on Answer Faithfulness."

    It did not. The judge decomposes the refusal sentence into one unsupported claim, so
    ``m == 1`` and ``answer_faithfulness``'s ``m == 0`` carve-out never fired — a model
    that refused in exactly the words prompt.py suggests was scored as maximally
    unfaithful, dragging corpus AF from 0.697 to 0.531 on the real seed set.
    """
    embedder, _, store, dense, sparse, _ = built
    retriever = HybridRetriever(store, dense, sparse, embedder)
    path = tmp_path / "questions.jsonl"
    path.write_text(
        '{"id":"unanswerable-001","question":"What is the insurance deductible?",'
        '"answer_type":"unanswerable","required_doc_ids":[],"relevant_doc_ids":[],'
        '"must_contain":{},"expected_synthesis_keys":[]}\n',
        encoding="utf-8",
    )
    settings = Settings(data_dir=tmp_path)
    refusal = FakeGenerator(reply="The provided context does not contain this information.")

    result = await run_eval(
        load_questions(path), retriever, refusal, LocalJudge(), settings, k=6
    )

    run = result.runs[0]
    assert run.af == pytest.approx(1.0), "a correct, uncited refusal is maximally faithful"


@pytest.mark.anyio
async def test_a_refusal_phrase_attached_to_a_cited_answer_is_not_laundered(tmp_path, built) -> None:
    """The guard on the carve-out above. ``is_refusal`` uses ``search``, so it also
    matches a refusal sentence tacked onto a real answer; prompt.py calls that false
    positive the worse failure because it would launder an unsupported answer into a
    perfect score. An answer that cites sources is never treated as a pure refusal.
    """
    embedder, _, store, dense, sparse, _ = built
    retriever = HybridRetriever(store, dense, sparse, embedder)
    path = tmp_path / "questions.jsonl"
    path.write_text(
        '{"id":"unanswerable-001","question":"What is the insurance deductible?",'
        '"answer_type":"unanswerable","required_doc_ids":[],"relevant_doc_ids":[],'
        '"must_contain":{},"expected_synthesis_keys":[]}\n',
        encoding="utf-8",
    )
    settings = Settings(data_dir=tmp_path)
    launderer = FakeGenerator(
        reply="The deductible is EUR 50,000 [S1]. The context does not contain more detail."
    )

    result = await run_eval(
        load_questions(path), retriever, launderer, LocalJudge(), settings, k=6
    )

    assert result.runs[0].af < 1.0, "a cited, unsupported claim must not ride a refusal phrase to 1.0"


@pytest.mark.anyio
async def test_a_correct_refusal_does_not_drag_marker_compliance_below_its_floor(
    tmp_path, built
) -> None:
    """The companion to the AF carve-out above, and a worse bug than it looks.

    A correct refusal cites nothing, so its per-answer marker compliance is 0.0 on a
    metric about citing. Averaged in, that caps the attainable corpus rate at 0.833 with
    one unanswerable question in six — so a *single* genuinely dropped-marker answer
    lands at 0.667, trips MARKER_COMPLIANCE_FLOOR, and makes report.py declare Cite, XDR
    and every CUR column unreliable because the model refused correctly. Measured: the
    2026-10-13 three-arm ablation flagged two of three arms exactly this way.

    So the refusal's compliance is None — undefined, like `cp` for the same question —
    and the rate is taken over the answers where the question is defined.
    """
    embedder, _, store, dense, sparse, _ = built
    retriever = HybridRetriever(store, dense, sparse, embedder)
    path = tmp_path / "questions.jsonl"
    path.write_text(
        '{"id":"single-001","question":"What is the statutory notification deadline?",'
        '"answer_type":"single_document","required_doc_ids":["gdpr"],'
        '"relevant_doc_ids":["gdpr"],"must_contain":{"gdpr":["72 hours"]},'
        '"expected_synthesis_keys":["72 hours"]}\n'
        '{"id":"unanswerable-001","question":"What is the insurance deductible?",'
        '"answer_type":"unanswerable","required_doc_ids":[],"relevant_doc_ids":[],'
        '"must_contain":{},"expected_synthesis_keys":[]}\n',
        encoding="utf-8",
    )
    settings = Settings(data_dir=tmp_path)

    class Refuser(FakeGenerator):
        """Cites properly when it can, refuses when it cannot — i.e. behaves."""

        async def complete(self, messages, *, temperature, max_tokens):
            question = messages[-1].content
            self.reply = (
                "The provided context does not contain this information."
                if "deductible" in question
                else "The controller must notify within 72 hours [S1]."
            )
            return await super().complete(
                messages, temperature=temperature, max_tokens=max_tokens
            )

    result = await run_eval(
        load_questions(path), retriever, Refuser(), LocalJudge(), settings, k=6
    )

    answered, refused = result.runs[0], result.runs[1]
    assert answered.marker_compliance == 1.0
    assert refused.marker_compliance is None, "undefined, not a failure to cite"

    # The rate is over the one defined answer, so a well-behaved model reads 1.0 — not
    # the 0.5 it would read if the refusal counted as a dropped marker.
    assert result.marker_compliance == pytest.approx(1.0)
    assert result.marker_compliance_n == 1
    assert result.markers_reliable is True

    rendered = report_mod.render(result)
    assert "1 correct refusal(s) excluded as undefined" in rendered
    assert "over 1/2 answer(s)" in rendered
    assert "UNRELIABLE" not in rendered


@pytest.mark.anyio
async def test_a_dropped_marker_still_fails_the_floor(tmp_path, built) -> None:
    """The guard on the exclusion. Excluding refusals must not make the floor
    unreachable — an answer that *should* have cited and did not is still a failure."""
    embedder, _, store, dense, sparse, _ = built
    retriever = HybridRetriever(store, dense, sparse, embedder)
    path = tmp_path / "questions.jsonl"
    path.write_text(
        '{"id":"single-001","question":"What is the statutory notification deadline?",'
        '"answer_type":"single_document","required_doc_ids":["gdpr"],'
        '"relevant_doc_ids":["gdpr"],"must_contain":{"gdpr":["72 hours"]},'
        '"expected_synthesis_keys":["72 hours"]}\n',
        encoding="utf-8",
    )
    settings = Settings(data_dir=tmp_path)
    # Answers the question, cites nothing, and is not a refusal.
    sloppy = FakeGenerator(reply="The controller must notify within 72 hours.")

    result = await run_eval(
        load_questions(path), retriever, sloppy, LocalJudge(), settings, k=6
    )
    assert result.runs[0].marker_compliance == 0.0, "an uncited real answer is a failure"
    assert result.marker_compliance == pytest.approx(0.0)
    assert result.markers_reliable is False
    assert "UNRELIABLE" in report_mod.render(result)


@pytest.mark.anyio
async def test_a_refusal_that_cites_is_not_excluded_either(tmp_path, built) -> None:
    """Same laundering guard as the AF carve-out, and deliberately the same condition in
    the code: an answer that cites sources is never treated as a pure refusal, so it
    cannot buy its way out of the compliance denominator by appending a refusal phrase.
    """
    embedder, _, store, dense, sparse, _ = built
    retriever = HybridRetriever(store, dense, sparse, embedder)
    path = tmp_path / "questions.jsonl"
    path.write_text(
        '{"id":"unanswerable-001","question":"What is the insurance deductible?",'
        '"answer_type":"unanswerable","required_doc_ids":[],"relevant_doc_ids":[],'
        '"must_contain":{},"expected_synthesis_keys":[]}\n',
        encoding="utf-8",
    )
    settings = Settings(data_dir=tmp_path)
    launderer = FakeGenerator(
        reply="The deductible is EUR 50,000 [S1]. The context does not contain more detail."
    )

    result = await run_eval(
        load_questions(path), retriever, launderer, LocalJudge(), settings, k=6
    )
    assert result.runs[0].marker_compliance == 1.0, "it cited, so it is measured"
    assert result.marker_compliance_n == 1


@pytest.mark.anyio
async def test_a_run_of_nothing_but_refusals_reports_no_rate_rather_than_zero(
    tmp_path, built
) -> None:
    embedder, _, store, dense, sparse, _ = built
    retriever = HybridRetriever(store, dense, sparse, embedder)
    path = tmp_path / "questions.jsonl"
    path.write_text(
        '{"id":"unanswerable-001","question":"What is the insurance deductible?",'
        '"answer_type":"unanswerable","required_doc_ids":[],"relevant_doc_ids":[],'
        '"must_contain":{},"expected_synthesis_keys":[]}\n',
        encoding="utf-8",
    )
    settings = Settings(data_dir=tmp_path)
    refusal = FakeGenerator(reply="The provided context does not contain this information.")

    result = await run_eval(
        load_questions(path), retriever, refusal, LocalJudge(), settings, k=6
    )
    assert result.marker_compliance is None
    assert result.marker_compliance_n == 0
    rendered = report_mod.render(result)
    assert "marker_compliance: n/a" in rendered
    # It must not claim the markers were checked and found good.
    assert "[OK]" not in rendered
    # And a comparison including such a run must not crash on `None < float`.
    assert "n/a" not in report_mod.render_comparison(
        [json.loads(json.dumps(asdict(result), default=str))], ["refusals-only"]
    )


@pytest.mark.anyio
async def test_a_refusal_with_the_required_document_in_context_scores_zero(
    tmp_path, built
) -> None:
    """The case that made corpus AF gameable, and it is not hypothetical.

    `single-002` ("What distinguishes a data controller from a data processor?") is an
    *easy* single-document question whose required doc is gdpr. Under dense-only and
    sparse-only, gdpr was retrieved — CP 0.74 and 1.00 — and the generator still replied
    "The provided context does not contain this information." The old carve-out keyed on
    "does this text look like a refusal" and scored it 1.00, twice, in two independent
    runs. A model that refuses everything therefore scored a corpus AF of 1.000.
    """
    embedder, _, store, dense, sparse, _ = built
    retriever = HybridRetriever(store, dense, sparse, embedder)
    path = tmp_path / "questions.jsonl"
    path.write_text(
        '{"id":"single-001","question":"What is the statutory notification deadline?",'
        '"answer_type":"single_document","required_doc_ids":["gdpr"],'
        '"relevant_doc_ids":["gdpr"],"must_contain":{"gdpr":["72 hours"]},'
        '"expected_synthesis_keys":["72 hours"]}\n',
        encoding="utf-8",
    )
    settings = Settings(data_dir=tmp_path)
    refuser = FakeGenerator(reply="The provided context does not contain this information.")

    result = await run_eval(
        load_questions(path), retriever, refuser, LocalJudge(), settings, k=6
    )
    run = result.runs[0]
    assert "gdpr" in run.retrieved_doc_ids, "the required doc really was in context"
    assert run.af == pytest.approx(0.0), "it declined with the material in hand"


@pytest.mark.anyio
async def test_a_refusal_without_the_required_document_is_still_correct(
    tmp_path, built
) -> None:
    """The other side of the rule, and the reason it is not simply "refusing an
    answerable question is 0.0". When retrieval fails to surface the required document,
    abstaining is exactly what the generator should do — inventing an answer would be
    the failure. That failure is retrieval's, and Cov and CP already carry it; charging
    it to AF as well would turn a generation metric into a retrieval one.
    """
    embedder, _, store, dense, sparse, _ = built
    retriever = HybridRetriever(store, dense, sparse, embedder)
    path = tmp_path / "questions.jsonl"
    # Requires a document that is not in the mini corpus at all, so it cannot be
    # retrieved however good the retriever is.
    path.write_text(
        '{"id":"single-003","question":"What is the cyber-liability deductible?",'
        '"answer_type":"single_document","required_doc_ids":["insurance-policy"],'
        '"relevant_doc_ids":["insurance-policy"],"must_contain":{},'
        '"expected_synthesis_keys":[]}\n',
        encoding="utf-8",
    )
    settings = Settings(data_dir=tmp_path)
    refuser = FakeGenerator(reply="The provided context does not contain this information.")

    result = await run_eval(
        load_questions(path), retriever, refuser, LocalJudge(), settings, k=6
    )
    run = result.runs[0]
    assert "insurance-policy" not in run.retrieved_doc_ids
    assert run.af == pytest.approx(1.0), "abstaining on missing context is correct"


@pytest.mark.anyio
async def test_refusing_every_question_no_longer_scores_a_perfect_corpus_af(
    tmp_path, built
) -> None:
    """The gameability check, stated as the property that matters: a generator that
    answers nothing must not win the benchmark."""
    embedder, _, store, dense, sparse, _ = built
    retriever = HybridRetriever(store, dense, sparse, embedder)
    path = tmp_path / "questions.jsonl"
    path.write_text(QUESTIONS, encoding="utf-8")   # xdr-001 + single-001, both answerable
    settings = Settings(data_dir=tmp_path)
    refuser = FakeGenerator(reply="The provided context does not contain this information.")

    result = await run_eval(
        load_questions(path), retriever, refuser, LocalJudge(), settings, k=8
    )
    assert result.af == pytest.approx(0.0), "refusing everything answerable scores 0.0"
    # And marker compliance is undefined throughout rather than a flood of failures:
    # the refusals are punished by AF, which is the metric that measures answering.
    assert result.marker_compliance is None
    assert result.marker_compliance_n == 0
