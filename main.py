import time
import logging
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from news_collector.graph import graph  # Adjust this import based on your actual file structure

# Setup logging to see the execution in your terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_news_workflow():
    """Function that triggers the LangGraph execution."""
    logger.info("Starting scheduled news collection task...")
    try:
        # Define the initial state for your graph
        initial_state = {
            "messages": [("user", "Collect and summarize today's top news.")],
            # Add other state keys your graph requires here
        }
        
        # Execute the graph
        result = graph.invoke(initial_state)
        
        logger.info("News collection completed successfully.")
        # Optional: Log a snippet of the result
        # print(result.get("summary", "No summary generated"))
        
    except Exception as e:
        logger.error(f"Error during news collection: {e}")

if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    
    # Schedule the job to run every day at 06:00
    scheduler.add_job(
        run_news_workflow, 
        'cron', 
        hour=6, 
        minute=0,
        id='daily_news_job'
    )
    
    scheduler.start()
    logger.info("Scheduler started. News collector will run daily at 06:00 AM.")

    # Keep the main thread alive
    try:
        while True:
            time.sleep(10)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler shut down.")
