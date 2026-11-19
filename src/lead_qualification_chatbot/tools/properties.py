from langchain_core.tools import BaseTool, tool

from lead_qualification_chatbot.data.repositories import PropertyRepository
from lead_qualification_chatbot.domain.messages import money
from lead_qualification_chatbot.domain.models import Property
from lead_qualification_chatbot.tools.common import NO_MATCH


def format_record(p: Property) -> str:
    return (
        f"Name: {p.property_name}\n"
        f"Address: {p.address}, {p.town}, {p.postcode}\n"
        f"Category: {p.listing_category.value}\n"
        f"Type: {p.property_type}; bedrooms: {p.bedrooms}; bathrooms: {p.bathrooms}\n"
        f"Price: {money(p.price_gbp)}\n"
        f"Status: {p.status.value}\n"
        f"Branch: {p.branch}; listed: {p.listed_date.isoformat()}\n"
        f"Description: {p.description[:300]}"
    )


def make_property_tool(repo: PropertyRepository) -> BaseTool:
    @tool
    async def lookup_property(query: str) -> str:
        """Look up property listings by name, address or id and return their details
        (type, bedrooms, price, status, branch). Input: the property name or address
        as the visitor wrote it. Several matches are returned if the name is ambiguous.
        - Call the tool once. Search again only if the first result was clearly about something \
            else, and never more than twice."""
        found = repo.find(query[:200])
        if not found:
            return NO_MATCH
        header = f"{len(found)} matching listing(s):\n\n"
        return header + "\n\n".join(format_record(p) for p in found)

    return lookup_property
