from flask import Flask, request, jsonify
import subprocess
import uuid
import os

app = Flask(__name__)


@app.route("/run", methods=["POST"])
def run_code():
    code = request.json.get("code", "")

    task_id = str(uuid.uuid4())
    os.makedirs(task_id, exist_ok=True)

    code_path = os.path.join(task_id, "main.py")

    with open(code_path, "w") as f:
        f.write(code)

    try:
        result = subprocess.run(
            ["python", code_path],
            capture_output=True,
            text=True,
            timeout=3,
        )
        output = result.stdout
        error = result.stderr
    except subprocess.TimeoutExpired:
        output = ""
        error = "TLE"

    return jsonify({"stdout": output, "stderr": error})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8111)
