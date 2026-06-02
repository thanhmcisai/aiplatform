"""
app.py — kc-admin-api: REST tối thiểu bọc Keycloak Admin API + UI quản lý user.

Endpoint:
  GET    /users                     — danh sách user (kèm vai trò)
  POST   /users                     — tạo user (body: username, email, password)
  DELETE /users/{user_id}           — xóa user
  POST   /users/{user_id}/password  — reset mật khẩu (body: password)
  GET    /roles                     — danh sách realm role
  POST   /users/{user_id}/roles     — gán role (body: role)
  DELETE /users/{user_id}/roles     — gỡ role (body: role)
  GET    /health

UI tại /  (static/index.html).

Lấy admin token bằng client admin-cli trên realm master, thao tác trên realm 'mlops'.
"""
import os

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

KC_URL = os.getenv("KC_URL", "http://keycloak:8080/auth")
KC_ADMIN = os.getenv("KC_ADMIN", "admin")
KC_ADMIN_PASSWORD = os.getenv("KC_ADMIN_PASSWORD", "admin_pw")
REALM = os.getenv("KC_REALM", "mlops")

app = FastAPI(title="Keycloak Admin API", description="REST tối thiểu bọc Keycloak")

_static = os.path.join(os.path.dirname(__file__), "static")
app.mount("/ui", StaticFiles(directory=_static), name="ui")


def token() -> str:
    r = requests.post(
        f"{KC_URL}/realms/master/protocol/openid-connect/token",
        data={"grant_type": "password", "client_id": "admin-cli",
              "username": KC_ADMIN, "password": KC_ADMIN_PASSWORD},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def hdr():
    return {"Authorization": f"Bearer {token()}"}


def admin(path: str) -> str:
    return f"{KC_URL}/admin/realms/{REALM}{path}"


class NewUser(BaseModel):
    username: str
    email: str = ""
    password: str
    first_name: str = ""
    last_name: str = ""


class PasswordBody(BaseModel):
    password: str


class RoleBody(BaseModel):
    role: str


@app.get("/")
def home():
    return FileResponse(os.path.join(_static, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/users")
def list_users():
    try:
        h = hdr()
        users = requests.get(admin("/users"), headers=h, timeout=30).json()
        out = []
        for u in users:
            roles = requests.get(
                admin(f"/users/{u['id']}/role-mappings/realm"),
                headers=h, timeout=30).json()
            role_names = [r["name"] for r in roles
                          if not r["name"].startswith("default-roles")]
            out.append({
                "id": u["id"],
                "username": u.get("username", ""),
                "email": u.get("email", ""),
                "firstName": u.get("firstName", ""),
                "lastName": u.get("lastName", ""),
                "roles": role_names,
            })
        return {"users": out}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Keycloak lỗi: {exc}")


@app.post("/users")
def create_user(body: NewUser):
    try:
        h = hdr()
        payload = {
            "username": body.username, "email": body.email,
            "firstName": body.first_name, "lastName": body.last_name,
            "enabled": True, "emailVerified": True,
            "credentials": [{"type": "password", "value": body.password,
                             "temporary": False}],
        }
        r = requests.post(admin("/users"), headers=h, json=payload, timeout=30)
        if r.status_code not in (201, 204):
            raise HTTPException(400, f"Không tạo được: {r.text}")
        return {"created": body.username}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Keycloak lỗi: {exc}")


@app.delete("/users/{user_id}")
def delete_user(user_id: str):
    r = requests.delete(admin(f"/users/{user_id}"), headers=hdr(), timeout=30)
    if r.status_code not in (200, 204):
        raise HTTPException(400, f"Không xóa được: {r.text}")
    return {"deleted": user_id}


@app.post("/users/{user_id}/password")
def reset_password(user_id: str, body: PasswordBody):
    r = requests.put(
        admin(f"/users/{user_id}/reset-password"), headers=hdr(),
        json={"type": "password", "value": body.password, "temporary": False},
        timeout=30)
    if r.status_code not in (200, 204):
        raise HTTPException(400, f"Reset thất bại: {r.text}")
    return {"reset": user_id}


@app.get("/roles")
def list_roles():
    roles = requests.get(admin("/roles"), headers=hdr(), timeout=30).json()
    return {"roles": [r["name"] for r in roles
                      if not r["name"].startswith("default-roles")]}


@app.post("/users/{user_id}/roles")
def assign_role(user_id: str, body: RoleBody):
    h = hdr()
    role = requests.get(admin(f"/roles/{body.role}"), headers=h, timeout=30).json()
    r = requests.post(admin(f"/users/{user_id}/role-mappings/realm"),
                      headers=h, json=[role], timeout=30)
    if r.status_code not in (200, 204):
        raise HTTPException(400, f"Gán role thất bại: {r.text}")
    return {"assigned": body.role}


@app.delete("/users/{user_id}/roles")
def remove_role(user_id: str, body: RoleBody):
    h = hdr()
    role = requests.get(admin(f"/roles/{body.role}"), headers=h, timeout=30).json()
    r = requests.delete(admin(f"/users/{user_id}/role-mappings/realm"),
                        headers=h, json=[role], timeout=30)
    if r.status_code not in (200, 204):
        raise HTTPException(400, f"Gỡ role thất bại: {r.text}")
    return {"removed": body.role}
