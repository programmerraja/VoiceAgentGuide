import asyncio
import logging
import os
from dotenv import load_dotenv

from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HelloWorldVoiceBot:
    def __init__(self):
        """Initialize the Hello World voice bot with all necessary services."""
        
        # Initialize AI services
        self.stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY")
        )
        
        self.llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo"  # Fast and cost-effective for this demo
        )
        
        self.tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # Friendly English voice
            language="en"
        )
        
        # Set up conversation context
        self.messages = [
            {
                "role": "system", 
                "content": """You are a friendly AI assistant named Pipecat Helper. 
                
                Your personality:
                - Enthusiastic about helping people learn Pipecat
                - Keep responses conversational and concise (1-2 sentences)
                - Always sound helpful and encouraging
                - If asked about Pipecat, explain it's a framework for building voice AI agents
                
                This is a 'Hello World' demo, so introduce yourself warmly and ask how you can help!"""
            }
        ]
        
        # Create context and aggregators for conversation management
        self.context = OpenAILLMContext(self.messages)
        self.context_aggregator = self.llm.create_context_aggregator(self.context)

    async def create_pipeline(self, transport):
        """Create and configure the processing pipeline."""
        
        pipeline = Pipeline([
            # Input: Receive audio from user
            transport.input(),
            
            # Speech-to-Text: Convert user's speech to text
            self.stt,
            
            # Context Management: Add user message to conversation history  
            self.context_aggregator.user(),
            
            # Language Model: Generate intelligent response
            self.llm,
            
            # Text-to-Speech: Convert response to natural speech
            self.tts,
            
            # Output: Send audio back to user
            transport.output(),
            
            # Context Management: Add bot response to conversation history
            self.context_aggregator.assistant(),
        ])
        
        return pipeline

    async def run_bot(self):
        """Main bot execution function."""
        
        try:
            # Create WebRTC transport for browser-based interaction
            transport = DailyTransport(
                "wss://api.daily.co/v1/rooms/temp",  # Temporary room
                None,  # No token needed for temp rooms
                "Pipecat Hello World Bot",
                DailyParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    vad_enabled=True,
                    vad_analyzer=SileroVADAnalyzer(),  # Voice activity detection
                    vad_audio_passthrough=True
                )
            )
            
            # Create the processing pipeline
            pipeline = await self.create_pipeline(transport)
            
            # Create pipeline task with configuration
            task = PipelineTask(
                pipeline,
                params=PipelineParams(
                    allow_interruptions=True,  # Enable natural conversation flow
                    enable_metrics=True,       # Track performance metrics
                    enable_usage_metrics=True  # Track API usage
                )
            )
            
            # Set up event handlers for user interactions
            @transport.event_handler("on_first_participant_joined")
            async def on_participant_joined(transport, participant):
                """When someone joins, greet them and start the conversation."""
                logger.info("Participant joined! Starting conversation...")
                
                # Add greeting instruction to context
                self.messages.append({
                    "role": "system", 
                    "content": "A user just joined! Greet them warmly and introduce yourself as Pipecat Helper. Ask how you can help them today!"
                })
                
                # Trigger the bot to speak
                await task.queue_frame(LLMMessagesFrame(self.messages))
            
            @transport.event_handler("on_participant_left") 
            async def on_participant_left(transport, participant, reason):
                """Handle user disconnection."""
                logger.info("Participant left. Ending conversation...")
                await task.queue_frame(EndFrame())
            
            @transport.event_handler("on_call_state_updated")
            async def on_call_state_updated(transport, state):
                """Handle call state changes."""
                if state == "left":
                    logger.info("Call ended.")
                    await task.queue_frame(EndFrame())
            
            # Start the pipeline runner
            runner = PipelineRunner()
            
            logger.info("🎙️ Hello World Voice Bot is starting up...")
            logger.info("🌐 Open http://localhost:7860 in your browser")
            logger.info("🎯 Click 'Connect' to start talking!")
            
            # Run the bot
            await runner.run(task)
            
        except Exception as e:
            logger.error(f"Error running bot: {e}")
            raise

# Entry point
async def main():
    """Main entry point for the Hello World bot."""
    bot = HelloWorldVoiceBot()
    await bot.run_bot()

if __name__ == "__main__":
    print("🎙️ Pipecat Hello World Voice Agent")
    print("==================================")
    print()
    print("This demo creates a friendly AI assistant that you can talk to!")
    print("The bot uses Deepgram for speech recognition, OpenAI for intelligence,")
    print("and Cartesia for natural speech synthesis.")
    print()
    print("Starting bot...")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Thanks for trying Pipecat!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. All required API keys in your .env file")
        print("2. Installed pipecat-ai with required dependencies")  
        print("3. Python 3.10 or later")