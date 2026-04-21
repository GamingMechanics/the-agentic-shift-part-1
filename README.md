# The Agentic Shift: Building Autonomous AI Agents with Google Cloud

This article was originally published on https://gamingmechanics.com/insights/the-agentic-shift-part-1

The current progress in generative AI has reached a critical inflexion point as we move from passive LLMs (stateless text generation) to active agents (stateful reasoning and execution), which is making generative AI more useful than ever before. I am certain you’ve been hearing all about agents all over the place as the new buzzword, and if you’ve spent any time on tech LinkedIn, X or Threads lately, it feels like “Chatbots” have been replaced by “Agents” overnight.

Behind that marketing gloss lies a profound architectural change in how we build software and, in this series of articles, we are going to move past the “Hello World” prompts, roll up our sleeves and build a production-grade, autonomous agentic system using the best of the Google ecosystem: Agent Development Kit (ADK), Vertex AI, Cloud Run and Firebase.

First, let’s have a look at what has changed in this new paradigm. In classic chatbots, our interaction has largely been linear: you send a prompt, the LLM predicts the next tokens, and you get a response, in a stateless, “read-only” experience. Agents have changed this game; they don’t just “talk”, they “reason” and “act”. They perceive a goal that can be broken down into a multi-step plan, select different tools for each task, and execute them until the goal is achieved, and they can constantly evaluate the state of completion of that goal.

For example, a traditional chatbot might tell you that the stock is low. An agent will check inventory, reorder items, notify suppliers and update your CRM, all without another prompt.

## The Stack

To build an agent that is more than just a toy or a prototype, you need three pillars:

1. The “brain” (Vertex AI): we need a model with a massive context window and high reasoning capabilities while keeping costs reasonable and being fast. As I write this (April 2026), Gemini 3 Flash is the gold standard for this, balancing lightning-fast latency with intelligence required to handle complex tool-calling, a 1-million-token context window and a score above 90% on GPQA Diamond.  
2. The “hands” (Cloud Run): our agents will need to interact with the real world to fetch live data, update databases, send messages and perform other tasks. Cloud Run provides the perfect serverless environment to host these tools securely and at scale. Agents can be quite “bursty”, and Cloud Run can handle the sudden spikes when an agent decides to call a bunch of tools at once.  
3. The “memory” (Firebase): our agent will only be as good as what it can remember, so we will look at how Firebase Data Connect can provide the relational persistence needed for long-term agent memory.

## The Series

Over these three articles, we are going to build a fully functional autonomous sales and inventory agent.

1. Part 1 (this article): the reasoning engine. We’ll explore the ReAct pattern, define our agent’s persona using the ADK, and set up our first reasoning loop.  
2. Part 2: the action layer. We will build and deploy custom API tools on Cloud Run, securing them with IAM so our agent can safely interact with private internal systems, and we will keep our agent updated with these tools.  
3. Part 3: persistence and scale. Finally, we’ll integrate Firebase to give our agent long-term memory and discuss how to monitor our agent’s thought process using Cloud Trace.

At the end of this series, you will be able to walk away with a blueprint for building agents that can safely and observably operate in real production environments.

## The ReAct Pattern: Bridging Reasoning and Acting

In the early days of generative AI, we relied on “chain of thought” prompting, where we would ask the model to think step-by-step in order to improve its logic. The issue was that these “thoughts” were trapped inside a text box. The ReAct framework (Reason + Act), introduced in 2022 and now a production standard, allows the model to interact with the world by interleaving reasoning traces with task-specific actions.

For our sales and inventory agent, the ReAct loop will follow a rigorous cycle:

1. Think: the model analyses the user’s request (for example, “we are running low on pink ribbons, please restock them”) and formulates a multi-step plan.  
2. Act: the agent then selects a tool from its toolkit, such as a Cloud Run-hosted inventory API that allows the agent to check how many pink ribbons are in stock, and generates the necessary parameters to call it.  
3. Observe: the agent receives the data back from the tool, for example, “current pink ribbon stock: 5, supplier lead time: 3 days”, and adds that information to its context window.

This loop repeats until the agent evaluates that the task is complete, such as notifying the user that the order has been placed and that your CRM has been updated.

In a little more detail, here is an example of how that loop would happen, step by step, once the agent received that initial request from the user:

* Thought: the user wants to restock pink ribbons. I first need to check the current inventory level to see how many to order.  
* Action: `inventory_manager.get_stock(item=”pink_ribbon”)`  
* Observation: `{"item": "pink_ribbon", "stock": 5, "reorder_point": 10}`  
* Thought: the current stock is 5, which is below the reorder level of 10, so I need to order more. I need to find the preferred supplier for this item.  
* Action: `inventory_manager.get_supplier(item=”pink_ribbon”)`  
* Observation: `{"item": "pink_ribbon", "supplier": “ACME”, "lead_time": “3 days”}`  
* Thought: I will now place an order for 20 units from ACME and update the CRM

Of course, this could get more complex depending on which tools you give your agent. We could allow the agent to see the typical sale pattern of pink ribbons to decide how many to order, for example, or allow the agent to check pricing with the supplier before placing the order if that tool were available. An agent is as complex as the tools you give it.

## Implementing the Reasoning Engine with Vertex AI ADK

To build this loop at an enterprise scale, we use the Vertex AI and the Agent Development Kit (ADK). ADK is an open source framework built by Google that manages the complexities of orchestration, allowing the developer to focus on the agent’s persona, logic, and available tools.

Using Vertex AI allows us to have managed infrastructure so that we don't have to manage the "orchestration loop" server ourselves; the Agent Engine handles the state. It is also secure by default with seamless integration with IAM, and it integrates natively with Google Cloud’s operations suite (Cloud Trace/Logging), which is crucial for debugging non-deterministic AI.

To start building an agent in Python using the ADK, we will first initialise a Python project using [uv](https://docs.astral.sh/uv/), with the following command:

`uv init the-agentic-shift-part-1 --python 3.13`

Then, change folders into the new project with `cd the-agentic-shift-part-1` and install the ADK with the following command:

`uv add google-adk`

Then, finally, run the following ADK command to create a new agent:

`adk create sales_inventory`

ADK will create a subfolder called `sales_inventory`, where an `agent.py` file will reside. Here is how we can define our reasoning engine in Python using the `google-adk` library, replacing the current contents of the `agent.py` file with the following:

```python
import os
from google import adk
import vertexai
from vertexai import agent_engines

# 1. Define which model we will use.  
# We use Flash for its 1M token context window and high reasoning scores.  
MODEL="gemini-3-flash-preview"

# Note: Vertex AI Agent Engine requires a different location from the `global` region required for the Gemini 3 Flash Preview above;  
# use a separate variable to avoid overriding the global location used by Gemini 3 Flash Preview above, as defined in the GOOGLE_CLOUD_LOCATION environment variable.  
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
3. OBSERVE: Analyse the tool output.  
4. REPEAT: Continue until the task is finalised.  
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
    # use a separate client to avoid overriding the global location used by Gemini 3 Flash Preview above, as defined in the GOOGLE_CLOUD_LOCATION environment variable.  
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
```

## Setting Up Google Cloud

In order to run our agent locally, as well as deploy it to Vertex AI Agent Engine, there are some settings to configure. It is somewhat out of the scope of this article to explain all of this setup, but I will quickly give you the required steps. For more details on setting up an ADK project, you can check the documentation on [https://adk.dev/get-started/python/](https://adk.dev/get-started/python/), and to see what deploying to Vertex AI should look like, the documentation is on [https://docs.cloud.google.com/agent-builder/agent-engine/quickstart-adk](https://docs.cloud.google.com/agent-builder/agent-engine/quickstart-adk). I also recommend checking out these docs: [https://adk.dev/agents/models/google-gemini/#google-cloud-vertex-ai](https://adk.dev/agents/models/google-gemini/#google-cloud-vertex-ai)

The first thing you will need to do is to create a project in your Google Cloud Console and enable billing. You will then need to create a bucket to be used as the staging bucket when deploying the agent to the Agent Engine. If you’re using the gcloud CLI, you can do that with the following command, replacing `the-name-of-your-bucket-here` with your own unique name, and perhaps selecting a different region if that suits you:

`gcloud storage buckets create gs://the-name-of-your-bucket-here --default-storage-class=STANDARD --location=EUROPE-WEST2 --uniform-bucket-level-access --public-access-prevention`

There are two constants in our code for these values, so replace them accordingly; they’re called `VERTEX_REGION` and `STAGING_BUCKET`.

In your Google Cloud Console, navigate to the Vertex AI dashboard ([https://console.cloud.google.com/vertex-ai/dashboard](https://console.cloud.google.com/vertex-ai/dashboard)) and select the “Enable all recommended APIs” option to enable the use of the Vertex AI APIs for your project.

Finally, edit or create the .env file inside the sales_inventory folder where our agent is, with the following values, replacing the project id in `GOOGLE_CLOUD_PROJECT` with the id of your own project:

```sh
GOOGLE_GENAI_USE_VERTEXAI=TRUE  
GOOGLE_CLOUD_PROJECT="your-project-id"  
GOOGLE_CLOUD_LOCATION=global
```

## Running and Deploying

With all those things in place, we can now test what we have done by running our agent locally with the command `adk run sales_inventory`, which will give us a prompt in the terminal to talk to our agent, or you can run the command `adk web`, which will provide us with a web UI on [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

To deploy our agent to Vertex AI Agent Engine, we simply run `python sales_inventory/agent.py` from the root folder of our project, and that will deploy it.

Please be aware that there are costs associated with both querying the agent when running locally and keeping an agent deployed. You can delete the deployed agent using the Google Cloud Console and going to Vertex AI and Agent Engine. Don’t forget to also delete the files in your Cloud Storage bucket.

At this point, our agent doesn’t do much, but you may get a response like this one if you say hi to it:

>Hello! I am **InventoryManager**, your Autonomous Sales & Inventory Agent.
>
>I’m here to help you keep your warehouse levels optimised. I can assist with:
>
>* Checking current inventory levels.  
>* Reviewing supplier information and lead times.  
>* Analysing sales data to predict stock needs.  
>* Creating purchase orders or restocking plans.  
>  How can I help you manage your warehouse today?

If you keep chatting, you’ll find that since it doesn’t have any tools, it won’t be capable of doing much. That’s what we will address in Part 2 of this series.

## Why Gemini 3 Flash?

In an agentic loop, latency is the enemy. Each "Think-Act-Observe" cycle adds a turn of inference time; if your model takes 10 seconds to "think," the user experience quickly degrades into frustration.

As of early 2026, Gemini 3 Flash has become the industry's "default" for autonomous agents for several reasons:

* **Speed & Latency**: It offers a first-token latency of roughly 0.21–0.37 seconds and an output speed of 163 tokens per second; nearly 3x faster than Gemini 3 Pro. This responsiveness is critical for real-time tool orchestration, where the agent may need to loop 5 or 6 times to solve a complex query.  
* **Pro-Grade Reasoning**: Despite being a "Flash" model, it punches well above its weight. It rivals frontier models on PhD-level reasoning benchmarks, scoring 90.4% on GPQA Diamond and an incredible 99.7% on AIME 2025 (with code execution enabled).  
* **Coding Superiority**: Interestingly, for agentic workflows, Flash actually outperforms many larger models on the SWE-bench Verified coding benchmark (78.0%), making it exceptionally reliable at generating the precise JSON parameters required for tool calling.  
* **Massive Context**: Its 1-million-token context window is a game-changer for inventory agents. You can pass entire warehouse logs or supplier contract PDFs directly into the prompt without worrying about the agent "losing the thread" or suffering from mid-conversation amnesia.  
* **Economic Scale**: At $0.50 per 1 million input tokens, it is less than a quarter of the cost of its "Pro" sibling. For autonomous agents running high-volume background tasks, this makes the difference between a project that is a "cool demo" and one that is "production-viable."

## Conclusion

In this first part of our series, we have laid the intellectual foundation for our autonomous system. We’ve moved beyond the "chatbot" mentality and embraced the ReAct pattern, a framework that allows our AI to think, act, and observe in a continuous loop.

By leveraging Google’s Agent Development Kit (ADK) and the Vertex AI Agent Engine, we’ve successfully deployed a "Brain" that is ready to reason. However, a brain without hands is just a dreamer. Currently, if you ask our InventoryManager to actually order those ribbons, it will acknowledge the task but ultimately hit a wall because it lacks the permission to touch the outside world.

In Part 2: The Action Layer, we are going to fix that. We will dive deep into Cloud Run to build secure, OpenAPI-compliant tools that our agent can call. We’ll cover how to use Identity-Aware Proxy (IAP) and IAM to ensure that our agent has the "Least Privilege" necessary to interact with your private business data safely.

Stay tuned. It will soon be time to give our agent some hands.

GitHub repo: https://github.com/GamingMechanics/the-agentic-shift-part-1/
