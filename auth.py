"""
Proteção por senha simples para o app.

Variáveis de ambiente:
  APP_PASSWORD  senha de acesso (obrigatória no Railway)
  SECRET_KEY    chave para assinar o cookie de sessão (defina uma string longa e aleatória)

Rodando localmente sem APP_PASSWORD, o login fica desativado.
No Railway (RAILWAY_ENVIRONMENT definido), sem APP_PASSWORD o app bloqueia o acesso.
"""

import hmac
import os
import secrets
from datetime import timedelta

from flask import jsonify, redirect, render_template_string, request, session, url_for

_LOGIN_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Acesso — {{ app_name }}</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;background:#0f172a;
         min-height:100vh;display:flex;align-items:center;justify-content:center;padding:16px}
    .card{background:#fff;border-radius:12px;padding:32px 28px;width:100%;max-width:360px;
          box-shadow:0 10px 30px rgba(0,0,0,.35)}
    h1{font-size:1.15rem;color:#0f172a;margin-bottom:4px}
    p{font-size:.85rem;color:#64748b;margin-bottom:20px}
    label{display:block;font-size:.75rem;font-weight:600;color:#475569;margin-bottom:6px}
    input{width:100%;padding:10px 12px;border:1px solid #cbd5e1;border-radius:8px;font-size:.95rem}
    input:focus{outline:none;border-color:#2563eb;box-shadow:0 0 0 3px rgba(37,99,235,.2)}
    button{margin-top:16px;width:100%;padding:10px;border:0;border-radius:8px;background:#2563eb;
           color:#fff;font-weight:600;font-size:.95rem;cursor:pointer}
    button:hover{background:#1d4ed8}
    .err{margin-top:12px;font-size:.8rem;color:#b91c1c}
  </style>
</head>
<body>
  <form class="card" method="post">
    <h1>{{ app_name }}</h1>
    <p>Acesso restrito. Digite a senha para continuar.</p>
    <label for="senha">Senha</label>
    <input id="senha" name="senha" type="password" autofocus autocomplete="current-password" required />
    <input type="hidden" name="next" value="{{ next_url }}" />
    <button type="submit">Entrar</button>
    {% if erro %}<div class="err">{{ erro }}</div>{% endif %}
  </form>
</body>
</html>"""


def init_auth(app, app_name: str) -> None:
    password = os.getenv("APP_PASSWORD", "")
    on_railway = bool(os.getenv("RAILWAY_ENVIRONMENT"))

    app.secret_key = os.getenv("SECRET_KEY") or secrets.token_hex(32)
    app.config.update(
        PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=on_railway,
    )

    @app.route("/login", methods=["GET", "POST"])
    def login():
        next_url = request.values.get("next") or "/"
        if not next_url.startswith("/") or next_url.startswith("//"):
            next_url = "/"
        erro = None
        if request.method == "POST":
            tentativa = request.form.get("senha", "")
            if password and hmac.compare_digest(tentativa.encode(), password.encode()):
                session.clear()
                session.permanent = True
                session["autenticado"] = True
                return redirect(next_url)
            erro = "Senha incorreta."
        return render_template_string(_LOGIN_HTML, app_name=app_name, next_url=next_url, erro=erro)

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.before_request
    def _exigir_login():
        if request.endpoint in ("login", "static"):
            return None
        if not password:
            if on_railway:
                return ("APP_PASSWORD não configurada no Railway. Defina a variável e reinicie o serviço.", 503)
            return None  # uso local sem senha
        if session.get("autenticado"):
            return None
        if request.path.startswith("/api/"):
            return jsonify({"error": "Sessão expirada. Recarregue a página e entre de novo."}), 401
        return redirect(url_for("login", next=request.full_path.rstrip("?")))
