"""Dataset information endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/datasets")
async def get_datasets():
    """Get metadata for all available datasets."""
    from backend.main import app_state

    datasets = []

    if app_state["metr_la_info"]:
        info = app_state["metr_la_info"].to_dict()
        info["description"] = "207 loop detectors on LA County highways. 5-minute traffic speed data."
        datasets.append(info)

    if app_state.get("india_mp_info"):
        info = app_state["india_mp_info"].to_dict()
        info["description"] = "250 sensors mapped across Madhya Pradesh cities (Bhopal, Indore, Gwalior, Jabalpur, Ujjain)."
        datasets.append(info)

    if app_state["pems_bay_info"]:
        info = app_state["pems_bay_info"].to_dict()
        info["description"] = "325 sensors in the Bay Area (CalTrans PeMS). Continuous traffic speed data."
        datasets.append(info)

    if app_state["uber_info"]:
        info = app_state["uber_info"].to_dict()
        info["description"] = "Aggregated and anonymized travel-time data between urban zones."
        datasets.append(info)

    return {"datasets": datasets, "demo_mode": app_state["demo_mode"]}
