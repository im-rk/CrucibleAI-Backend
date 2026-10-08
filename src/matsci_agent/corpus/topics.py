"""Topics the shipped corpus covers. Each becomes one targeted arXiv query (cond-mat.mtrl-sci)."""

TOPICS: list[str] = [
    # energy storage
    "solid-state electrolyte ionic conductivity",
    "garnet LLZO lithium",
    "sulfide solid electrolyte argyrodite",
    "lithium-ion battery cathode",
    "sodium-ion battery electrode",
    "polymer electrolyte",
    # photovoltaics and optoelectronics
    "perovskite solar cell bandgap",
    "halide perovskite stability",
    "wide bandgap semiconductor GaN",
    "ultrawide bandgap AlGaN",
    "gallium oxide Ga2O3",
    "deep ultraviolet LED",
    # functional materials
    "thermoelectric figure of merit",
    "superconductor critical temperature",
    "high entropy alloy",
    "two-dimensional transition metal dichalcogenide bandgap",
    "MXene",
    "ferroelectric dielectric",
    "topological insulator",
    # catalysis and porous materials
    "photocatalyst water splitting",
    "metal-organic framework gas adsorption",
    # computation and data
    "machine learning materials property prediction",
    "density functional theory band gap",
    "high-throughput screening materials discovery",
    "materials database Materials Project",
    "large language model materials science",
]


def build_query(topic: str) -> str:
    """arXiv query: every keyword of the topic (AND), restricted to the materials-science category.

    An exact phrase is too strict for arXiv's index (hyphens, word order), so keywords are AND-ed.
    """
    words = [w for w in topic.replace("-", " ").split() if w.lower() not in {"and", "of", "the"}]
    return "cat:cond-mat.mtrl-sci AND " + " AND ".join(f"all:{w}" for w in words)
