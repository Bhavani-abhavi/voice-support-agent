"""Mock CRM + billing system.

Invoice dates are generated relative to today, so the demo scenarios behave the
same whichever day a reviewer opens the app (the duplicate charge is always 3 days
old, the annual charge is always outside its 30-day window, and so on).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date, timedelta


@dataclass
class Invoice:
    id: str
    date: date
    amount: float
    description: str
    status: str = "paid"  # paid | refunded | partially_refunded
    refunded: float = 0.0
    duplicate_of: str | None = None

    def public(self) -> dict:
        return {
            "invoice_id": self.id,
            "date": self.date.isoformat(),
            "amount_usd": self.amount,
            "description": self.description,
            "status": self.status,
            "refunded_usd": self.refunded,
            "flag": f"possible duplicate of {self.duplicate_of}" if self.duplicate_of else None,
        }


@dataclass
class Customer:
    id: str
    name: str
    email: str
    zip_code: str
    tier: str  # free | pro | business
    billing_cycle: str | None  # monthly | annual | contract
    contract_status: str  # active | past_due | cancelled
    renewal_date: date | None
    seats: int | None = None
    account_manager: str | None = None
    billing_email: str = ""
    notifications: str = "all"
    auto_renew: bool = True
    invoices: list[Invoice] = field(default_factory=list)

    def profile(self) -> dict:
        return {
            "customer_id": self.id,
            "name": self.name,
            "tier": self.tier,
            "billing_cycle": self.billing_cycle,
            "contract_status": self.contract_status,
            "renewal_date": self.renewal_date.isoformat() if self.renewal_date else None,
            "seats": self.seats,
            "account_manager": self.account_manager,
            "auto_renew": self.auto_renew,
        }


def build_customers(today: date | None = None) -> dict[str, Customer]:
    t = today or date.today()
    d = lambda days: t - timedelta(days=days)  # noqa: E731
    customers = [
        Customer(
            id="CUS-1001", name="Alex Rivera", email="alex.rivera@example.com", zip_code="64111",
            tier="pro", billing_cycle="monthly", contract_status="active",
            renewal_date=d(3) + timedelta(days=30), billing_email="alex.rivera@example.com",
            invoices=[
                Invoice("INV-1043", d(3), 12.00, "Pro monthly", duplicate_of="INV-1042"),
                Invoice("INV-1042", d(3), 12.00, "Pro monthly"),
                Invoice("INV-0987", d(33), 12.00, "Pro monthly"),
            ],
        ),
        Customer(
            id="CUS-2002", name="Priya Shah", email="priya.shah@example.com", zip_code="94107",
            tier="business", billing_cycle="contract", contract_status="active",
            renewal_date=d(-120), seats=40, account_manager="Dana Kim",
            billing_email="finance@shahdesign.example",
            invoices=[
                Invoice("INV-2201", d(10), 12000.00, "Business annual contract, 40 seats"),
                Invoice("INV-2202", d(10), 300.00, "Onboarding fee (billed in error)"),
            ],
        ),
        Customer(
            id="CUS-3003", name="Sam Lee", email="sam.lee@example.com", zip_code="10001",
            tier="pro", billing_cycle="annual", contract_status="past_due",
            renewal_date=d(-330), billing_email="sam.lee@example.com",
            invoices=[Invoice("INV-3100", d(35), 120.00, "Pro annual")],
        ),
        Customer(
            id="CUS-4004", name="Jordan Kim", email="jordan.kim@example.com", zip_code="60614",
            tier="free", billing_cycle=None, contract_status="active", renewal_date=None,
            billing_email="jordan.kim@example.com",
        ),
    ]
    return {c.email: c for c in customers}


DEMO_ACCOUNTS = [
    {"name": "Alex Rivera", "email": "alex.rivera@example.com", "zip": "64111", "tier": "Pro monthly",
     "scenario": "Charged twice this month: full tool chain ending in a refund"},
    {"name": "Priya Shah", "email": "priya.shah@example.com", "zip": "94107", "tier": "Business",
     "scenario": "$300 fee billed in error (within limit); a $12,000 refund needs a human"},
    {"name": "Sam Lee", "email": "sam.lee@example.com", "zip": "10001", "tier": "Pro annual, past due",
     "scenario": "Wants a refund 35 days after renewing: policy says no"},
    {"name": "Jordan Kim", "email": "jordan.kim@example.com", "zip": "60614", "tier": "Free",
     "scenario": "Sync problems and plan questions"},
]


class MockBackend:
    """Per-session copy of the CRM, so one reviewer's refund doesn't change another's demo."""

    def __init__(self, today: date | None = None) -> None:
        self.today = today or date.today()
        self.customers = copy.deepcopy(build_customers(self.today))
        self.refund_seq = 5000
        self.ticket_seq = 7000

    def find(self, email: str) -> Customer | None:
        return self.customers.get(email.strip().lower())

    def next_refund_id(self) -> str:
        self.refund_seq += 1
        return f"RF-{self.refund_seq}"

    def next_ticket_id(self) -> str:
        self.ticket_seq += 1
        return f"TCK-{self.ticket_seq}"
