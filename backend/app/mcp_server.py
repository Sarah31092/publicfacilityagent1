from mcp.server.fastmcp import FastMCP

from app.services.mcp_tools import get_facility_details, search_facilities

mcp = FastMCP("public-facility-search")


@mcp.tool()
def search_public_facilities(
    query: str = "",
    area: str = "전국",
    mode: str = "structured",
    limit: int = 10,
    facility_type: str = "",
    fee: str = "전체",
    operating: str = "전체",
    facility_name: str = "",
    place_name: str = "",
) -> dict:
    """Search public facilities by region, name/type, or vector similarity."""
    return search_facilities(
        query=query,
        area=area,
        mode=mode,
        limit=limit,
        facility_type=facility_type,
        fee=fee,
        operating=operating,
        facility_name=facility_name,
        place_name=place_name,
    )


@mcp.tool()
def get_public_facility_details(facility_id: str) -> dict:
    """Get all available details for a facility returned by search."""
    return get_facility_details(facility_id)


if __name__ == "__main__":
    mcp.run()
