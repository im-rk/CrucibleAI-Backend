"""Service layer entry point for running the research workflow."""

# from __future__ import annotations

# from matsci_agent.agents.orchestrator import initialize_state
# from matsci_agent.domain.schemas import ResearchRequest


# class ResearchService:
#     """Coordinates agent execution and persistence."""

#     async def run(self, request: ResearchRequest) -> dict[str, object]:
#         state = initialize_state(request.query)
#         return {"state": state, "max_results": request.max_results}
