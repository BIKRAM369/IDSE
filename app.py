"""
app.py ─ Flask REST API for Industrial Predictive Maintenance Web App
Run with:  python app.py
Open:      http://localhost:5000
"""

import os
from flask import Flask, jsonify, render_template
from flask_cors import CORS
import analysis

app = Flask(__name__,
            static_folder="static",
            template_folder="templates")
CORS(app)


def ok(data):
    return jsonify({"success": True, "data": data})


def err(msg, code=500):
    return jsonify({"success": False, "error": str(msg)}), code


# ── Frontend ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── Status ────────────────────────────────────────────────────────────────────
@app.route("/api/status")
def api_status():
    return jsonify({
        "status":          "ok",
        "message":         "Industrial Predictive Maintenance API",
        "pipeline_status": analysis.get_status(),
    })


# ── Individual steps ──────────────────────────────────────────────────────────
@app.route("/api/run/load",       methods=["POST"])
def api_load():
    try: return ok(analysis.run_load())
    except Exception as e: return err(e)

@app.route("/api/run/preprocess", methods=["POST"])
def api_preprocess():
    try: return ok(analysis.run_preprocess())
    except Exception as e: return err(e)

@app.route("/api/run/eda",        methods=["POST"])
def api_eda():
    try: return ok(analysis.run_eda())
    except Exception as e: return err(e)

@app.route("/api/run/split",      methods=["POST"])
def api_split():
    try: return ok(analysis.run_split())
    except Exception as e: return err(e)

@app.route("/api/run/train",      methods=["POST"])
def api_train():
    try: return ok(analysis.run_train())
    except Exception as e: return err(e)

@app.route("/api/run/tune",       methods=["POST"])
def api_tune():
    try: return ok(analysis.run_tune())
    except Exception as e: return err(e)

@app.route("/api/run/evaluate",   methods=["POST"])
def api_evaluate():
    try: return ok(analysis.run_evaluate())
    except Exception as e: return err(e)

@app.route("/api/run/features",   methods=["POST"])
def api_features():
    try: return ok(analysis.run_features())
    except Exception as e: return err(e)

@app.route("/api/run/errors",     methods=["POST"])
def api_errors():
    try: return ok(analysis.run_errors())
    except Exception as e: return err(e)

@app.route("/api/run/all",        methods=["POST"])
def api_run_all():
    try: return ok(analysis.run_all())
    except Exception as e: return err(e)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🏭  Industrial Predictive Maintenance Platform")
    print("  📡  API  → http://localhost:5000/api/status")
    print("  🌐  App  → http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000, host="0.0.0.0")
