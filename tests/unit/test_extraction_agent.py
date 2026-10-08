"""Unit tests for ExtractionAgent (fake LLM, fake Materials Project client)."""

from __future__ import annotations

from matsci_agent.agents.nodes.extraction import (
    ExtractionAgent,
    ExtractionOutput,
    RawRecord,
    heuristic_extract,
)
from matsci_agent.domain.schemas import MaterialProperty
from matsci_agent.domain.state import initial_state
from matsci_agent.infrastructure.llm.anthropic_client import LLMError


def raw(**overrides) -> RawRecord:
    base = dict(
        passage=0,
        material_name="LLZO", material_formula="Li7La3Zr2O12", property_name="ionic conductivity",
        value_raw="0.8 mS/cm", value_numeric=0.8, unit="mS/cm", conditions="25 C",
        confidence="high", excerpt="LLZO reaches 0.8 mS/cm.",
    )
    return RawRecord(**{**base, **overrides})


class FakeLLM:
    def __init__(self, records=None, available=True, error: Exception | None = None):
        self.available = available
        self.records = records or []
        self.error = error

    async def structured(self, prompt, schema, **kwargs):
        if self.error:
            raise self.error
        return ExtractionOutput(records=self.records)


class FakeMP:
    def __init__(self, online=True, props=None):
        self.online = online
        self.props = props or {}
        self.asked: list[str] = []

    def get_properties(self, formula):
        self.asked.append(formula)
        return self.props.get(formula, [])


def chunk(cid="p1_0", text="LLZO shows a high ionic conductivity.", arxiv="p1"):
    return {
        "id": cid, "text": text, "relevance_score": 0.9,
        "metadata": {"arxiv_id": arxiv, "title": "Garnet paper", "doi": "10.1/abc"},
    }


def state_with(*chunks):
    state = initial_state("goal")
    state["retrieved_papers"] = list(chunks)
    return state


class Events:
    def __init__(self):
        self.seen = []

    async def __call__(self, event):
        self.seen.append(event)

    @property
    def config(self):
        return {"configurable": {"emit": self}}


async def test_llm_records_are_normalised_and_carry_provenance() -> None:
    agent = ExtractionAgent(FakeLLM([raw()]), FakeMP(online=True))
    out = await agent.run(state_with(chunk()))

    rec = out["extracted_properties"][0]
    assert rec["property_name"] == "ionic_conductivity"
    assert rec["unit"] == "S/cm"
    assert rec["value_numeric"] == 0.8e-3  # 0.8 mS/cm -> S/cm
    assert rec["material_formula"] == "Li7La3Zr2O12"
    assert (rec["source_title"], rec["source_doi"], rec["source_id"]) == ("Garnet paper", "10.1/abc", "p1")
    assert rec["chunk_id"] == "p1_0"


async def test_numeric_value_is_recovered_from_raw_text_when_llm_omits_it() -> None:
    record = raw(value_raw="3.4 × 10⁻³ S/cm", value_numeric=None, unit="S/cm")
    out = await ExtractionAgent(FakeLLM([record]), FakeMP()).run(state_with(chunk()))
    assert out["extracted_properties"][0]["value_numeric"] == 3.4e-3


async def test_invalid_formulas_are_dropped_but_the_record_is_kept() -> None:
    record = raw(material_formula="LLZO")
    out = await ExtractionAgent(FakeLLM([record]), FakeMP()).run(state_with(chunk()))
    rec = out["extracted_properties"][0]
    assert rec["material_formula"] is None and rec["material_name"] == "LLZO"


async def test_duplicate_records_from_overlapping_chunks_collapse() -> None:
    out = await ExtractionAgent(FakeLLM([raw()]), FakeMP()).run(
        state_with(chunk("p1_0"), chunk("p1_1"))
    )
    literature = [r for r in out["extracted_properties"] if r["method"] == "literature"]
    assert len(literature) == 1


async def test_materials_project_records_are_added_for_found_formulas() -> None:
    mp = FakeMP(online=True, props={"Li7La3Zr2O12": [MaterialProperty(
        material_id="mp-1", formula="Li7La3Zr2O12", property_name="bandgap", value=4.1, unit="eV",
        source="Materials Project", doi="10.1038/x",
    )]})
    events = Events()
    out = await ExtractionAgent(FakeLLM([raw()]), mp).run(state_with(chunk()), events.config)

    db = [r for r in out["extracted_properties"] if r["method"] != "literature"]
    assert mp.asked == ["Li7La3Zr2O12"]
    assert db[0]["method"] == "Materials Project (DFT)" and db[0]["confidence"] == "high"
    assert db[0]["source_id"] == "mp-1" and db[0]["value_numeric"] == 4.1
    assert [e["event"] for e in events.seen] == ["extraction_start", "mp_lookup", "properties_extracted"]


async def test_offline_database_is_labelled_and_warned_about() -> None:
    mp = FakeMP(online=False, props={"GaN": [MaterialProperty(
        material_id="offline-reference", formula="GaN", property_name="bandgap", value=3.4, unit="eV",
    )]})
    out = await ExtractionAgent(FakeLLM([raw(material_formula="GaN", material_name="GaN")]), mp).run(
        state_with(chunk())
    )
    db = [r for r in out["extracted_properties"] if r["method"] != "literature"]
    assert db[0]["method"] == "offline reference" and db[0]["confidence"] == "medium"
    assert any("MATERIALS_PROJECT_API_KEY" in w for w in out["warnings"])


async def test_a_failing_chunk_is_skipped_and_reported_not_fatal() -> None:
    class Flaky(FakeLLM):
        async def structured(self, prompt, schema, **kwargs):
            if "BOOM" in prompt:
                raise RuntimeError("parser exploded")
            return await super().structured(prompt, schema, **kwargs)

    out = await ExtractionAgent(Flaky([raw()]), FakeMP(), batch_size=1).run(
        state_with(chunk("a_0", "fine", "a"), chunk("b_0", "BOOM", "b"))
    )
    assert any("Skipped chunk b_0" in w for w in out["warnings"])
    assert any(r["chunk_id"] == "a_0" for r in out["extracted_properties"])


async def test_llm_error_on_a_chunk_falls_back_to_heuristics() -> None:
    text = "The garnet Li7La3Zr2O12 reaches an ionic conductivity of 1.2 × 10⁻³ S/cm at room temperature."
    out = await ExtractionAgent(FakeLLM(error=LLMError("down")), FakeMP()).run(state_with(chunk(text=text)))
    rec = out["extracted_properties"][0]
    assert rec["confidence"] == "low" and rec["value_numeric"] == 1.2e-3
    assert any("fell back to heuristics" in w for w in out["warnings"])


async def test_no_chunks_yields_empty_result_and_warning() -> None:
    out = await ExtractionAgent(FakeLLM(), FakeMP()).run(state_with())
    assert out["extracted_properties"] == []
    assert any("no relevant" in w.lower() for w in out["warnings"])


def test_heuristic_extract_reads_common_sentence_patterns() -> None:
    text = (
        "The garnet Li7La3Zr2O12 reaches an ionic conductivity of 3 mS cm⁻¹ at room temperature. "
        "GaN has a band gap of 3.4 eV. The electrochemical stability window is 5.2 V for this cell."
    )
    records = {(r.property_name, r.material_formula): r for r in heuristic_extract(chunk(text=text))}
    assert records[("ionic_conductivity", "Li7La3Zr2O12")].value_raw.startswith("3 ")
    assert records[("bandgap", "GaN")].value_raw == "3.4 eV"
    # no formula in that sentence: attributed to the most recent one before it
    assert ("electrochemical_stability_window", "GaN") in records
    assert all(r.confidence == "low" for r in records.values())


async def test_passages_are_batched_into_one_call_and_records_keep_their_own_source() -> None:
    class Counting(FakeLLM):
        calls = 0

        async def structured(self, prompt, schema, **kwargs):
            Counting.calls += 1
            assert "[0] Paper: Paper A" in prompt and "[1] Paper: Paper B" in prompt
            return ExtractionOutput(records=[
                raw(passage=1, material_formula="GaN", material_name="GaN", property_name="bandgap",
                    value_raw="3.4 eV", value_numeric=3.4, unit="eV"),
                raw(passage=0),
                raw(passage=7),  # a passage that does not exist
            ])

    a = {**chunk("a_0", arxiv="a"), "metadata": {"arxiv_id": "a", "title": "Paper A", "doi": ""}}
    b = {**chunk("b_0", arxiv="b"), "metadata": {"arxiv_id": "b", "title": "Paper B", "doi": ""}}
    out = await ExtractionAgent(Counting(), FakeMP(online=True)).run(state_with(a, b))

    assert Counting.calls == 1
    lit = {r["material_formula"]: r for r in out["extracted_properties"] if r["method"] == "literature"}
    assert (lit["GaN"]["chunk_id"], lit["GaN"]["source_title"]) == ("b_0", "Paper B")
    assert (lit["Li7La3Zr2O12"]["chunk_id"], lit["Li7La3Zr2O12"]["source_id"]) == ("a_0", "a")
    assert any("citing a passage that does not exist" in w for w in out["warnings"])


async def test_twelve_chunks_take_three_calls() -> None:
    class Counting(FakeLLM):
        calls = 0

        async def structured(self, prompt, schema, **kwargs):
            Counting.calls += 1
            return ExtractionOutput(records=[])

    await ExtractionAgent(Counting(), FakeMP()).run(state_with(*[chunk(f"p{i}_0", arxiv=f"p{i}") for i in range(12)]))
    assert Counting.calls == 3


async def test_latex_formatting_from_abstracts_is_cleaned_before_validation() -> None:
    record = raw(material_name="Li$_{7}$La$_{3}$Zr$_{2}$O$_{12}$", material_formula="Li$_{7}$La$_{3}$Zr$_{2}$O$_{12}$")
    out = await ExtractionAgent(FakeLLM([record]), FakeMP()).run(state_with(chunk()))
    rec = out["extracted_properties"][0]
    assert rec["material_formula"] == "Li7La3Zr2O12" and rec["material_name"] == "Li7La3Zr2O12"
