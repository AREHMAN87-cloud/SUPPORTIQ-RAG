from langchain.tools import tool
from search import hybrid_search, rerank
from datetime import datetime, timedelta
import random

# Fake order data standing in for a real orders database
FAKE_ORDERS = {
    "PK-39281": {"status": "In Transit", "eta": "2 days", "courier": "TCS"},
    "PK-45981": {"status": "Delivered", "delivered_on": "2026-08-20"},
    "PK-10234": {"status": "Processing", "eta": "3-4 days"},
}

@tool
def search_knowledge_base(question: str) -> str:
    """Search company policies and FAQs for general questions about returns, shipping, warranty, and account issues."""
    candidates = hybrid_search(question, k=20)
    top_chunks = rerank(question, candidates, top_n=5)
    context = "\n\n".join(text for _, text, _ in top_chunks)  # 3 values, not 4
    return context if context else "No relevant information found in the knowledge base."""

@tool
def get_order_status(order_id: str) -> str:
    """Look up the current status of a specific order by its order ID (e.g. PK-39281)."""
    order = FAKE_ORDERS.get(order_id.upper())
    if not order:
        return f"No order found with ID {order_id}. Please double-check the order number."
    return str(order)


@tool
def check_return_eligibility(order_id: str) -> str:
    """Check whether a specific order is still eligible for return based on delivery date and the 14-day return policy."""
    order = FAKE_ORDERS.get(order_id.upper())
    if not order:
        return f"No order found with ID {order_id}."
    if order.get("status") != "Delivered":
        return f"Order {order_id} hasn't been delivered yet (current status: {order['status']}), so return eligibility doesn't apply yet."

    delivered_on = datetime.strptime(order["delivered_on"], "%Y-%m-%d")
    days_since = (datetime.now() - delivered_on).days

    if days_since <= 14:
        return f"Order {order_id} was delivered {days_since} days ago and IS eligible for return (within the 14-day window)."
    else:
        return f"Order {order_id} was delivered {days_since} days ago and is NOT eligible for return (past the 14-day window)."

@tool
def create_support_ticket(order_id: str, issue: str) -> str:
    """Create a support ticket for an issue that couldn't be resolved automatically, such as a complaint, damaged item, or unresolved dispute."""
    ticket_id = f"TICKET-{random.randint(10000, 99999)}"
    return f"Support ticket {ticket_id} created for order {order_id}. Issue: '{issue}'. Our team will follow up within 24 hours."
