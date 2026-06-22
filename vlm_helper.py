import json
import base64
import time
import urllib.request
import urllib.error
import logging
from pathlib import Path

# Default setting to enable/disable VLM request logging
ENABLE_VLM_LOGGING = True

# Setup logging to both console and a local log file
log_file_path = Path(__file__).resolve().parent / "vlm_requests.log"
logger = logging.getLogger("vlm_helper")
logger.setLevel(logging.INFO if ENABLE_VLM_LOGGING else logging.WARNING)

def set_logging_enabled(enabled: bool) -> None:
    """Dynamically enable or disable informational VLM logging."""
    global ENABLE_VLM_LOGGING
    ENABLE_VLM_LOGGING = enabled
    logger.setLevel(logging.INFO if enabled else logging.WARNING)

# Avoid adding multiple handlers if the module is reloaded
if not logger.handlers:
    # File handler
    fh = logging.FileHandler(log_file_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)

def _post_request(url: str, payload: dict) -> dict:
    """Perform a POST request using urllib."""
    model = payload.get("model", "unknown-model")
    messages = payload.get("messages", [])
    num_images = 0
    total_image_bytes = 0
    if messages:
        images_list = messages[0].get("images", [])
        num_images = len(images_list)
        # Estimate base64-decoded sizes for diagnostics
        total_image_bytes = sum(len(img) * 3 // 4 for img in images_list)

    payload_bytes = len(json.dumps(payload).encode("utf-8"))
    logger.info(
        f"POST {url} | Model: {model} | Images: {num_images} "
        f"(~{total_image_bytes // 1024} KB decoded) | Payload: {payload_bytes // 1024} KB"
    )

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            res_data = response.read().decode("utf-8")
            elapsed = time.time() - t0
            logger.info(
                f"POST {url} | Success | {elapsed:.1f}s elapsed | "
                f"Response length: {len(res_data)} chars"
            )
            return json.loads(res_data)
    except urllib.error.URLError as e:
        elapsed = time.time() - t0
        error_details = ""
        if hasattr(e, "read"):
            try:
                error_details = " | Details: " + e.read().decode("utf-8").strip()
            except Exception:
                pass
        err_msg = f"Ollama server connection failed: {e}{error_details}"
        logger.error(f"POST {url} | Failed after {elapsed:.1f}s: {err_msg}")
        raise RuntimeError(f"Failed: {err_msg}")
    except Exception as e:
        elapsed = time.time() - t0
        err_msg = f"Request failed: {e}"
        logger.error(f"POST {url} | Failed after {elapsed:.1f}s: {err_msg}")
        raise RuntimeError(err_msg)

def _get_request(url: str) -> dict:
    """Perform a GET request using urllib."""
    logger.info(f"GET {url}")
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            res_data = response.read().decode("utf-8")
            logger.info(f"GET {url} | Success")
            return json.loads(res_data)
    except urllib.error.URLError as e:
        err_msg = f"Ollama server connection failed: {e}"
        logger.error(f"GET {url} | Failed: {err_msg}")
        raise RuntimeError(err_msg)
    except Exception as e:
        err_msg = f"Request failed: {e}"
        logger.error(f"GET {url} | Failed: {err_msg}")
        raise RuntimeError(err_msg)

def check_ollama_status(server_url: str) -> dict:
    """
    Check if Ollama server is running and fetch available models.
    Returns a dict with status details.
    """
    server_url = server_url.rstrip("/")
    tags_url = f"{server_url}/api/tags"
    try:
        data = _get_request(tags_url)
        models = [m["name"] for m in data.get("models", [])]
        return {
            "status": "connected",
            "models": models,
            "error": None
        }
    except Exception as e:
        return {
            "status": "disconnected",
            "models": [],
            "error": str(e)
        }

def _to_base64(image_bytes: bytes) -> str:
    """Convert raw image bytes to a base64 encoded string."""
    return base64.b64encode(image_bytes).decode("utf-8")

def describe_figure(image_bytes: bytes, model: str, server_url: str) -> str:
    """Generate a caption/description of an image using Ollama VLM."""
    server_url = server_url.rstrip("/")
    chat_url = f"{server_url}/api/chat"

    logger.info(f"[describe_figure] Image size: {len(image_bytes) // 1024} KB | Model: {model}")
    img_b64 = _to_base64(image_bytes)

    prompt = (
        "Analyze this figure extracted from a document. Describe it in detail, "
        "explaining any charts, diagrams, flowcharts, or visual patterns. "
        "Summarize its key information concisely in 1-2 paragraphs. "
        "Do not include conversational boilerplate (like 'Sure, here is...'); start describing directly."
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [img_b64]
            }
        ],
        "stream": False,
        "options": {
            "num_ctx": 16384
        }
    }

    try:
        res = _post_request(chat_url, payload)
        content = res.get("message", {}).get("content", "").strip()
        logger.info(f"[describe_figure] Caption generated ({len(content)} chars)")
        return content
    except Exception as e:
        logger.error(f"[describe_figure] Failed: {e}")
        raise RuntimeError(f"Failed to generate figure caption: {e}")

def summarize_table(csv_str: str, image_bytes: bytes | None, model: str, server_url: str) -> str:
    """Generate a summary of a table's data and optionally its visual layout."""
    server_url = server_url.rstrip("/")
    chat_url = f"{server_url}/api/chat"

    img_kb = (len(image_bytes) // 1024) if image_bytes else 0
    logger.info(
        f"[summarize_table] CSV length: {len(csv_str)} chars | "
        f"Image: {'yes (' + str(img_kb) + ' KB)' if image_bytes else 'none'} | Model: {model}"
    )

    prompt = (
        f"Analyze the following table data (extracted in CSV format):\n\n{csv_str}\n\n"
        "Provide a concise summary of the key highlights, trends, and takeaways from this table "
        "in 1-2 paragraphs. Avoid conversational filler."
    )

    messages = [{"role": "user", "content": prompt}]

    # If vision model and image crop is provided, include it
    if image_bytes is not None:
        img_b64 = _to_base64(image_bytes)
        messages[0]["images"] = [img_b64]
        messages[0]["content"] += "\n\nRefer to the attached visual image crop of the table if needed to verify details."

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_ctx": 16384
        }
    }

    try:
        res = _post_request(chat_url, payload)
        content = res.get("message", {}).get("content", "").strip()
        logger.info(f"[summarize_table] Summary generated ({len(content)} chars)")
        return content
    except Exception as e:
        # If we failed and sent an image (likely due to non-vision model), retry as text-only
        if image_bytes is not None:
            logger.warning(f"[summarize_table] Vision call failed ({e}), retrying text-only...")
            messages[0].pop("images", None)
            messages[0]["content"] = messages[0]["content"].replace(
                "\n\nRefer to the attached visual image crop of the table if needed to verify details.", ""
            )
            try:
                res = _post_request(chat_url, payload)
                content = res.get("message", {}).get("content", "").strip()
                logger.info(f"[summarize_table] Text-only fallback succeeded ({len(content)} chars)")
                return content
            except Exception as retry_e:
                logger.error(f"[summarize_table] Text-only fallback also failed: {retry_e}")
                raise RuntimeError(f"Failed to generate table summary (text fallback failed): {retry_e}")
        logger.error(f"[summarize_table] Failed: {e}")
        raise RuntimeError(f"Failed to generate table summary: {e}")

def enhance_table_structure(csv_str: str, image_bytes: bytes, model: str, server_url: str) -> str:
    """
    Passes a visual table crop and a draft CSV to the VLM, requesting structural corrections.
    Returns the corrected CSV string.
    """
    server_url = server_url.rstrip("/")
    chat_url = f"{server_url}/api/chat"

    logger.info(
        f"[enhance_table_structure] CSV length: {len(csv_str)} chars | "
        f"Image: {len(image_bytes) // 1024} KB | Model: {model}"
    )
    img_b64 = _to_base64(image_bytes)

    prompt = (
        "Analyze the table in this image and compare it with the following draft CSV extraction "
        "(which may have alignment, column grouping, merged cells, or OCR/spelling issues):\n\n"
        f"{csv_str}\n\n"
        "Please reconstruct the table accurately based on the visual layout. Ensure headers are correctly aligned "
        "with data rows. Output the fully corrected table ONLY as valid CSV, enclosed inside a standard markdown code block: "
        "```csv\n[corrected CSV here]\n```. Do not output any explanations or extra conversational text."
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [img_b64]
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,  # low temperature for structure extraction correctness
            "num_ctx": 16384
        }
    }

    try:
        res = _post_request(chat_url, payload)
        content = res.get("message", {}).get("content", "").strip()

        # Parse the CSV from code block
        if "```csv" in content:
            csv_part = content.split("```csv")[1].split("```")[0].strip()
            logger.info(f"[enhance_table_structure] Extracted CSV block ({len(csv_part)} chars)")
            return csv_part
        elif "```" in content:
            csv_part = content.split("```")[1].split("```")[0].strip()
            if csv_part.startswith("csv\n"):
                csv_part = csv_part[4:]
            logger.info(f"[enhance_table_structure] Extracted generic code block ({len(csv_part)} chars)")
            return csv_part
        logger.warning(f"[enhance_table_structure] No code block found in VLM response, returning raw content ({len(content)} chars)")
        return content
    except Exception as e:
        logger.error(f"[enhance_table_structure] Failed: {e}")
        raise RuntimeError(f"Failed to enhance table structure: {e}")

def get_model_capabilities(server_url: str, model_name: str) -> dict:
    """
    Query Ollama /api/show to check if the model supports vision/multimodal features.
    Returns a dict with capability details.
    """
    server_url = server_url.rstrip("/")
    show_url = f"{server_url}/api/show"
    try:
        data = _post_request(show_url, {"model": model_name})
        capabilities = data.get("capabilities", [])
        details = data.get("details", {})
        families = details.get("families", []) or [details.get("family", "")]

        has_vision = "vision" in capabilities or any(fam and "clip" in str(fam).lower() for fam in families)
        logger.info(
            f"[get_model_capabilities] {model_name}: vision={has_vision} | "
            f"capabilities={capabilities} | families={families}"
        )
        return {
            "has_vision": has_vision,
            "families": families,
            "capabilities": capabilities
        }
    except Exception as e:
        # Fallback to name heuristics in case of connection or metadata errors
        name_lower = model_name.lower()
        has_vision = "vision" in name_lower or "vl" in name_lower or "minicpm" in name_lower
        logger.warning(
            f"[get_model_capabilities] API call failed for '{model_name}' ({e}), "
            f"falling back to name heuristics: vision={has_vision}"
        )
        return {
            "has_vision": has_vision,
            "families": [],
            "capabilities": []
        }

def refine_merged_table(csv_str: str, image_bytes_list: list[bytes], model: str, server_url: str) -> str:
    """
    Passes a draft merged CSV and a list of cropped page/table fragment images to the VLM
    to resolve any column misalignment, row shifts, or header promotions.
    Returns the corrected CSV string.
    """
    server_url = server_url.rstrip("/")
    chat_url = f"{server_url}/api/chat"

    # Convert all images to base64
    images_b64 = [_to_base64(img_bytes) for img_bytes in image_bytes_list if img_bytes]
    total_img_kb = sum(len(b) // 1024 for b in image_bytes_list if b)
    logger.info(
        f"[refine_merged_table] CSV length: {len(csv_str)} chars | "
        f"Fragments: {len(images_b64)} images (~{total_img_kb} KB total) | Model: {model}"
    )

    prompt = (
        "You are an expert document data cleaning assistant.\n"
        "We have merged multiple consecutive table fragments from a PDF into a single draft CSV.\n"
        "However, during rule-based merging, some columns may have been misaligned, shifted, or headers may have leaked into data rows.\n\n"
        f"Here is the draft merged CSV:\n\n{csv_str}\n\n"
        "I have attached the visual cropped images of the table fragments in order.\n"
        "Using the visual layout, align all rows under the primary table headers and clean up any parsing errors.\n"
        "Output the fully corrected table ONLY as a valid CSV, enclosed inside a standard markdown code block:\n"
        "```csv\n[corrected CSV here]\n```\n"
        "Do not write any conversation, thoughts, or explanations."
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": images_b64
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,  # low temperature for structure extraction correctness
            "num_ctx": 32768  # 32k context size for multi-image table refinement!
        }
    }

    try:
        res = _post_request(chat_url, payload)
        content = res.get("message", {}).get("content", "").strip()

        # Parse the CSV from code block
        if "```csv" in content:
            csv_part = content.split("```csv")[1].split("```")[0].strip()
            logger.info(f"[refine_merged_table] Extracted CSV block ({len(csv_part)} chars)")
            return csv_part
        elif "```" in content:
            csv_part = content.split("```")[1].split("```")[0].strip()
            if csv_part.startswith("csv\n"):
                csv_part = csv_part[4:]
            logger.info(f"[refine_merged_table] Extracted generic code block ({len(csv_part)} chars)")
            return csv_part
        logger.warning(f"[refine_merged_table] No code block found in VLM response, returning raw content ({len(content)} chars)")
        return content
    except Exception as e:
        logger.error(f"[refine_merged_table] Failed: {e}")
        raise RuntimeError(f"Failed to refine merged table: {e}")
