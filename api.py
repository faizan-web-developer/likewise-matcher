from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import tempfile
import os
from gpt import process_ipads
from fastapi.responses import FileResponse

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_dataframe_for_json(df):
    """Clean DataFrame to ensure JSON serialization works."""
    def clean_value(val):
        if pd.isna(val) or pd.isnull(val):
            return None
        if isinstance(val, (np.float32, np.float64)):
            if np.isnan(val) or np.isinf(val):
                return None
            return float(val)
        if isinstance(val, (np.int64, np.int32)):
            return int(val)
        if isinstance(val, pd.Series):
            return clean_value(val.iloc[0] if len(val) > 0 else None)
        if isinstance(val, (list, np.ndarray)):
            return [clean_value(x) for x in val]
        return str(val)  # Convert all other types to strings for safety

    return [{k: clean_value(v) for k, v in row.items()} for row in df.to_dict('records')]

@app.post("/process")
async def process_files(
    products: UploadFile = File(...),
    likewise: UploadFile = File(...)
):
    """Process Excel files for product matching."""
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as products_tmp:
            products_content = await products.read()
            products_tmp.write(products_content)
            products_tmp_path = products_tmp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as likewise_tmp:
            likewise_content = await likewise.read()
            likewise_tmp.write(likewise_content)
            likewise_tmp_path = likewise_tmp.name

        try:
            # Read the Excel files into DataFrames
            products_df = pd.read_excel(products_tmp_path)
            likewise_df = pd.read_excel(likewise_tmp_path, sheet_name='iPad')  # Note the sheet name

            # Process the files using process_ipads
            result = process_ipads(likewise_df, products_df)
            
            if result is None:
                raise HTTPException(status_code=500, detail="Failed to process products")
            
            # Clean the DataFrame for JSON serialization
            cleaned_data = clean_dataframe_for_json(result)
            
            return {
                "status": "success",
                "data": cleaned_data
            }

        finally:
            # Clean up temporary files
            try:
                os.unlink(products_tmp_path)
                os.unlink(likewise_tmp_path)
            except:
                pass  # Ignore cleanup errors

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download the processed Excel file."""
    file_path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        file_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)