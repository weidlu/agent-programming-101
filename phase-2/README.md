## Phase 2 (LangGraph): Customer Service Agent

This folder contains a minimal LangGraph example that matches the "Phase 2" plan:

- State (Pydantic): `CustomerServiceState`
- Nodes: intent classification, human handoff, refund confirmation (interrupt), refund processing (tool)
- Human-in-the-loop: refund requires a "yes/no" confirmation before issuing
- Idempotency: refund node won't issue a second refund if re-run

### Run

```bash
cd /Users/weidlu/workspace/agent-programming-101
source .venv/bin/activate
python /Users/weidlu/workspace/agent-programming-101/phase-2/customer_service_agent.py
```

Try:

- `我要退款，订单号 123456`
- `我要退款，我很生气，订单号 123456` (will route to human handoff)

