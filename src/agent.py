import logging
import os
import json 
from dotenv import load_dotenv

from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, google, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("agent")

load_dotenv(".env")

# --- DEBUGGING BLOCK ---
print("--- Loading LiveKit credentials from .env ---")
print(f"LIVEKIT_URL: {os.environ.get('LIVEKIT_URL')}")
print(f"LIVEKIT_API_KEY: {os.environ.get('LIVEKIT_API_KEY')}")
print(f"LIVEKIT_API_SECRET: {'SET' if os.environ.get('LIVEKIT_API_SECRET') else 'NOT SET'}")
print("-----------------------------------------")

# --- DEFINE AGENT PERSONAS ---

class OverallAgent(Agent):
    """This is the general-purpose, friendly assistant."""
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

    @function_tool
    async def lookup_weather(self, context: RunContext, location: str):
        """Use this tool to look up current weather information in the given location."""
        logger.info(f"Looking up weather for {location}")
        return "sunny with a temperature of 70 degrees."

class CounsellorAgent(Agent):
    """This agent acts as an empathetic career counsellor."""
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly and empathetic career counsellor for Aestr Alpha.
            Your goal is to understand the user's background, their degree, and their career aspirations.
            Provide guidance on which Aestr Alpha programs, like 'Deep Cloud Engineering' or 'Fintech', would be the best fit for them.
            Keep your answers encouraging and supportive.""",
        )

class AdminAgent(Agent):
    """This agent provides factual, administrative information."""
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are an administrative assistant for Aestr Alpha.
            You provide factual information about program logistics.
            Answer questions about course duration (e.g., '6-month intensive track'), number of projects, class schedules, and how to apply.
            Your tone is professional, clear, and direct.""",
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # DEFINITIVE FIX: --- AGENT ROUTING LOGIC ---
    # The JobContext (ctx) contains the job information. The job itself
    # has a 'participant' attribute which holds the ParticipantInfo,
    # including the metadata we need.
    
    agent_mode = 'overall'  # Default mode
    
    # The correct attribute is ctx.job.participant
    participant = ctx.job.participant

    if participant and participant.metadata:
        try:
            # The metadata is a JSON string, so we parse it
            metadata = json.loads(participant.metadata)
            # We get the 'agent_mode' key we set in our frontend API
            agent_mode = metadata.get('agent_mode', 'overall')
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode metadata: {participant.metadata}")

    logger.info(f"initiating agent with mode: {agent_mode}")

    # Instantiate the correct agent based on the mode
    agent: Agent
    if agent_mode == 'counsellor':
        agent = CounsellorAgent()
    elif agent_mode == 'administration':
        agent = AdminAgent()
    else:
        agent = OverallAgent()

    session = AgentSession(
        llm=google.LLM(model="gemini-1.5-flash"),
        stt=deepgram.STT(model="nova-2", language="multi"),
        tts=cartesia.TTS(voice="6f84f4b8-58a2-430c-8c79-688dad597532"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # ... (Debugging and other event listeners remain the same) ...
    @session.on("vad_activity")
    def on_vad_activity(activity):
        print(f"[VAD DEBUG] Activity detected: type={activity.type.name}, confidence={activity.confidence}")

    @session.on("stt_transcript_received")
    def on_stt_transcript(interim: bool, text: str):
        transcript_type = "INTERIM" if interim else "FINAL"
        print(f"[STT DEBUG] Transcript received ({transcript_type}): {text}")
        
    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("false positive interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))