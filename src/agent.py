
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

# --- Main Agent Definition ---

class AestrAlphaAgent(Agent):
    """
    This is the primary AI assistant for Aestr Alpha.
    It is a helpful and friendly guide for prospective students.
    """
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful and friendly voice AI assistant for an educational platform called Aestr Alpha.
            You eagerly assist users with their questions about the platform, its courses, and career prospects.
            Your responses are concise, encouraging, and to the point.
            Avoid complex formatting or punctuation like emojis and asterisks.
            You are curious and have a positive and supportive tone.
            """,
        )

    @function_tool
    async def get_course_details(self, context: RunContext, course_name: str):
        """
        Use this tool to get information about a specific course offered by Aestr Alpha.
        Examples: 'Deep Cloud Engineering', 'Fintech', 'Robotics'.
        """
        logger.info(f"Looking up details for course: {course_name}")
        # In a real application, you would fetch this from a database or a file.
        if "cloud" in course_name.lower():
            return "The Deep Cloud & Multi-Cloud Engineering program is a 6-month intensive track with over 4 live production projects. It's our most popular course for landing high-impact roles."
        elif "fintech" in course_name.lower():
            return "The FinTech Engineering course is a specialized program focused on building the future of money. It covers everything from blockchain to high-frequency trading systems."
        else:
            return f"I can't seem to find specific details on {course_name} right now, but our programs are designed to turn degrees into high-paying careers."

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # --- AGENT INITIALIZATION (SIMPLIFIED) ---
    # We no longer need to check metadata. We will always use the main agent.
    logger.info("Initializing AestrAlphaAgent...")
    agent = AestrAlphaAgent()

    session = AgentSession(
        llm=google.LLM(model="gemini-2.5-flash"),
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
