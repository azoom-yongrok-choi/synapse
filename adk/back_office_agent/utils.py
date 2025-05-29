import json
import os
import logging
import traceback


# Recursively extract all field paths from data_type.json properties
def extract_field_paths(properties, parent=""):
    paths = []
    for k, v in properties.items():
        path = f"{parent}.{k}" if parent else k
        # If nested or object, go deeper
        if v.get("type") == "nested" or "properties" in v:
            child_props = v.get("properties", {})
            paths.extend(extract_field_paths(child_props, path))
        else:
            paths.append(path)
    return paths


# Extract only the paths corresponding to main fields (location, fee, security, space)
def get_default_parking_fields(data_type_path=None):
    if data_type_path is None:
        # Relative path from agent.py
        data_type_path = os.path.join(os.path.dirname(__file__), "data_type.json")
    with open(data_type_path, encoding="utf-8") as f:
        data_type = json.load(f)
    all_paths = extract_field_paths(data_type["properties"])
    # Main field keywords
    location_keywords = [
        "address",
        "addressView",
        "location",
        "city.name",
        "prefecture.name",
        "region.name",
        "nearbyStations.name",
    ]
    fee_keywords = [
        "payment.fee",
        "spaces.rent",
        "spaces.rentMin",
        "spaces.rentTaxClass",
        "referralFeeTotal",
        "storageDocument.issuingFee",
    ]
    security_keywords = ["securityFacilities.status", "spaces.facility"]
    space_keywords = ["spaces", "capacity", "hasDivisionDrawing"]

    # Filter only paths that match the keywords
    def match_keywords(path, keywords):
        for kw in keywords:
            if path.endswith(kw):
                return True
        return False

    default_fields = [
        p
        for p in all_paths
        if (
            match_keywords(p, location_keywords)
            or match_keywords(p, fee_keywords)
            or match_keywords(p, security_keywords)
            or match_keywords(p, space_keywords)
        )
    ]
    # Order: location, fee, security, space
    ordered = []
    for group in [location_keywords, fee_keywords, security_keywords, space_keywords]:
        for kw in group:
            for p in default_fields:
                if p.endswith(kw) and p not in ordered:
                    ordered.append(p)
    return ordered


# Recursively extract all nested field paths from data_type.json properties
def extract_nested_field_paths(properties, parent=""):
    nested_paths = []
    for k, v in properties.items():
        path = f"{parent}.{k}" if parent else k
        if v.get("type") == "nested":
            nested_paths.append(path)
        if "properties" in v:
            nested_paths.extend(extract_nested_field_paths(v["properties"], path))
    return nested_paths


# Get all nested field paths from data_type.json
def get_nested_fields(data_type_path=None):
    if data_type_path is None:
        data_type_path = os.path.join(os.path.dirname(__file__), "data_type.json")
    with open(data_type_path, encoding="utf-8") as f:
        data_type = json.load(f)
    return extract_nested_field_paths(data_type["properties"])


async def ensure_required_params_callback(tool, args, tool_context):
    logging.info(
        f"[TOOL GUARDRAIL] Called ensure_required_params_callback with tool={tool}, args={args}, tool_context={tool_context}"
    )
    try:
        required_params = getattr(tool, "required", []) or []
        missing_params = [
            p for p in required_params if p not in args or args[p] in (None, "")
        ]
        if missing_params:
            if "queryBody" in missing_params:
                user_message_en = (
                    "Your search is missing the required 'queryBody' parameter. "
                    "Please provide more details about your search (e.g., location, price, facility, space, etc.) so I can help you better!"
                )
                user_text = getattr(tool_context, "user_input", None)
                llm_agent = getattr(tool_context, "llm_agent", None)
                if llm_agent and user_text:
                    prompt = (
                        f"Translate the following message into the user's language, matching the tone and style of the user's last message.\n"
                        f"User's last message: {user_text}\n"
                        f"Message: {user_message_en}"
                    )
                    try:
                        response = await llm_agent.generate(prompt)
                        user_message = (
                            response.text
                            if hasattr(response, "text")
                            else str(response)
                        )
                    except Exception as e:
                        logging.error(f"[TOOL GUARDRAIL] LLM translation failed: {e}")
                        user_message = user_message_en
                else:
                    user_message = user_message_en
                return {"status": "error", "error_message": user_message}
            return {
                "status": "error",
                "error_message": f"Required information ({', '.join(missing_params)}) is missing. Please provide more details!",
            }
        logging.info(
            "[TOOL GUARDRAIL] All required params present. Tool execution allowed."
        )
        return None
    except Exception as e:
        logging.error(
            f"[TOOL GUARDRAIL] Exception in ensure_required_params_callback: {e}"
        )
        logging.error(traceback.format_exc())
        return {
            "status": "error",
            "error_message": f"Exception occurred during parameter check: {e}",
        }
