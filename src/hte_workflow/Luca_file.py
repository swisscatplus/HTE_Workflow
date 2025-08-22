import os
import time


from shared.APIArksuite import APIArksuite
from models.workflow.catalog import Catalog
from models.workflow.workflow import Workflow
from models.run.run import Run
from dotenv import load_dotenv
from shared.file_management import *

# Load environment variables from .env file
load_dotenv(os.path.join("config", ".env"))

PROJECT_FOLDER = os.getenv("PROJECT_FOLDER")
WORKFLOWSTEPS_FOLDER = os.path.join(PROJECT_FOLDER, os.getenv("WORKFLOWSTEPS_FOLDER"))
DATA_WORKFLOW_FOLDER = os.path.join(PROJECT_FOLDER, os.getenv("DATA_WORKFLOW_FOLDER"))
WORKFLOW_FOLDER = os.path.join(PROJECT_FOLDER, os.getenv("WORKFLOW_FOLDER"))
CATALOG_FOLDER = os.path.join(PROJECT_FOLDER, os.getenv("CATALOG_FOLDER"))

def main():
    start_time = time.time()
    # Authenticate only if no token is stored
    # The authentication is mandatory when access to ArkSuite via API!

    Auth = APIArksuite()
    Auth.authenticate()

    # Retrieve all workflowsteps/workflows and stor their separated_and_detailed version
    obj_ark1 = APIArksuite()
    obj_ark2 = APIArksuite()
    obj_ark1.get_all_workflowsteps(separated_and_detailed=True)
    obj_ark2.get_all_workflows(separated_and_detailed=True)

    # Retrieve all run status then get statistics if the runs are Completed, CompletedWithErrors, or Canceled
    obj_run1 = Run()
    obj_run1.get_multiple_run_statistics()




    """
    Test creation 
    """

    """
    Catalogs : "Unique"--> generate the UniqueMethods_catalog.json from get_all_workflowsteps.json, "Full" regenerate the UniqueMethods_catalog.json, and generate the Full_catalog
    """
    catalog = Catalog()
    # catalog.create_catalogs("Unique")
    catalog.create_catalogs("Full")
    """
    Workflow/workflowsteps related functions
    """

    # Input
    # delete_all_json_files("new_workflow")

    workflow = Workflow()
    workflow.create_new_workflow(
        "20250819_Morandi_H2_Racemic_1 5 (Catscreen)",
        True,
        "Set Pressure Block",
        "Wait",
        "Stop Pressure Block"
    )


    """
    Runs related functions
    """

    obj_run = Run()
    obj_run.get_multiple_run_statistics()


    # obj_run.get_multiple_run_status()

    end_time = time.time()  # End the timer
    execution_time = end_time - start_time # Calculate duration
    print(f"🚀 Execution time: {execution_time:.3f} seconds")
if __name__ == "__main__":
    main()
