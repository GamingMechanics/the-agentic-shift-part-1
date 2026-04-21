import os
from google import adk
import vertexai
from vertexai import agent_engines

# 1. Define which model we will use.
# We use Flash for its 1M token context window and high reasoning scores.
MODEL="gemini-3-flash-preview"

# Note: Vertex AI Agent Engine requires a different location from the `global` region required for the Gemini 3 Flash Preview above;
# use a separate variable to avoid overriding the global location used by Gemini 3 Flash Preview above as defined in the GOOGLE_CLOUD_LOCATION environment variable.
VERTEX_REGION="europe-west2"

# Agent Engine also requires a staging bucket for temporary files during deployment.
# Make sure to create this bucket in the same region as your Agent Engine deployment to avoid latency and egress costs.
STAGING_BUCKET="gs://the-name-of-your-bucket-here"

# 2. Define the Agent's Persona (System Instructions)
# This acts as the "operating system" for the ReAct loop.
SYSTEM_INSTRUCTIONS = """
You are an Autonomous Sales & Inventory Agent. 
Your goal is to ensure warehouse levels are optimal. 
When a user makes a request:
1. THINK: What data do I need? (Inventory, Supplier lists, CRM).
2. ACT: Call the appropriate tools. 
3. OBSERVE: Analyze the tool output.
4. REPEAT: Continue until the task is finalized.
"""

# 3. Create the Agent using ADK
# This handles the state management and orchestration.
root_agent = adk.Agent(
    name="InventoryManager",
    model=MODEL,
    instruction=SYSTEM_INSTRUCTIONS,
)

# 4. Deploy to Vertex AI Agent Engine
# Agent Engine provides a managed runtime to scale our agent.
# Run this file directly (`python agent.py`) to deploy; never runs on import.
if __name__ == "__main__":
    app = agent_engines.AdkApp(agent=root_agent)
    # Agent Engine requires a different location from the `global` region required for the Gemini 3 Flash Preview above;
    # use a separate client to avoid overriding the global location used by Gemini 3 Flash Preview above as defined in the GOOGLE_CLOUD_LOCATION environment variable.
    agent_engine_client = vertexai.Client(
        project=os.environ.get('GOOGLE_CLOUD_PROJECT'),
        location=VERTEX_REGION,
    )
    remote_agent = agent_engine_client.agent_engines.create(
        agent=app,
        config={
            "requirements": [
                "google-cloud-aiplatform[agent_engines,adk]",
                "pydantic",
                "cloudpickle",
            ],
            "staging_bucket": STAGING_BUCKET,
        },
    )
