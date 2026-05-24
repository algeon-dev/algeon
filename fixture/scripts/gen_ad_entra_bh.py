"""Generate a synthetic AD + Entra hybrid BloodHound fixture as Parquet.

Implements the AD/Entra half of the first slice described in
`docs/e2e/2026-05-19-bloodhound-graph-fixture-plan.md`. Emits typed wide
tables for both AD (SharpHound-shaped) and Entra (AzureHound-shaped) data
plus an AD+Entra slice of the generic `bh_nodes` / `bh_edges` layer. A
small fraction of AD users are linked to AZ users via an `ADSyncedTo` edge
to model AAD Connect identity sync — this is the hybrid bridge for path
workloads that span on-prem and cloud.

Tables emitted:
  ad_users
  ad_groups
  ad_computers
  ad_aces
  ad_sessions
  ad_cert_templates
  azure_users
  azure_groups
  azure_apps
  azure_service_principals
  azure_role_assignments
  bh_nodes        (AD+Entra slice)
  bh_edges        (AD+Entra slice, including ADSyncedTo hybrid bridge)

Wide-table column choices are guided by `bloodhound-docs` sample data:
SharpHound user/group/computer Properties bags and AzureHound payloads.
Only a high-selectivity subset is materialized — full SharpHound/AzureHound
fidelity is out of scope for the first slice.

Env:
  BH_ADENTRA_SCALE     bh_sf1 (default)
  BH_ADENTRA_OUT_DIR   output root; default <repo>/fixture/data/ad_entra/<scale>/parquet
  BH_ADENTRA_SEED      override deterministic seed (default derived from scale)

`bh_sf1` target shape:
  AD:    2 domains, 5,000 users, 500 groups, 800 computers,
         ~50k ACEs, ~2,000 sessions, 20 cert templates.
  Entra: 1 tenant, 3 subscriptions, 10 resource groups, 3,000 users,
         300 groups, 500 apps, 500 service principals, 200 VMs,
         ~5,000 role assignments.
  Hybrid: ~30% of AD users mapped to AZ users via ADSyncedTo.

Run:
  fixture/scripts/run_gen.sh ad_entra
  BH_ADENTRA_SCALE=bh_sf1 fixture/scripts/run_gen.sh ad_entra
"""

import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ScaleParams:
    name: str
    seed: int
    # AD half.
    n_domains: int
    n_ad_users: int
    n_ad_groups: int
    n_ad_computers: int
    n_aces: int
    n_sessions: int
    n_cert_templates: int
    # Entra half.
    n_tenants: int
    n_subscriptions: int
    n_resource_groups: int
    n_az_users: int
    n_az_groups: int
    n_az_apps: int
    n_az_service_principals: int
    n_az_vms: int
    n_az_role_assignments: int
    # Hybrid bridge.
    ad_sync_fraction: float


SCALE_PRESETS: dict[str, ScaleParams] = {
    "bh_sf1": ScaleParams(
        name="bh_sf1",
        seed=20260519,
        n_domains=2,
        n_ad_users=5_000,
        n_ad_groups=500,
        n_ad_computers=800,
        n_aces=50_000,
        n_sessions=2_000,
        n_cert_templates=20,
        n_tenants=1,
        n_subscriptions=3,
        n_resource_groups=10,
        n_az_users=3_000,
        n_az_groups=300,
        n_az_apps=500,
        n_az_service_principals=500,
        n_az_vms=200,
        n_az_role_assignments=5_000,
        ad_sync_fraction=0.30,
    ),
}


ACE_RIGHTS = ("GenericAll", "GenericWrite", "WriteDacl", "WriteOwner", "Owns", "AllExtendedRights", "AddMember")
ACE_RIGHT_WEIGHTS = np.array([0.10, 0.15, 0.10, 0.10, 0.05, 0.15, 0.35])
AZ_BUILTIN_ROLES = (
    "Owner",
    "Contributor",
    "Reader",
    "User Access Administrator",
    "Virtual Machine Administrator Login",
    "Virtual Machine Contributor",
    "Key Vault Administrator",
    "Storage Blob Data Owner",
)
AZ_ROLE_WEIGHTS = np.array([0.10, 0.30, 0.30, 0.05, 0.05, 0.10, 0.05, 0.05])


def _power_law_weights(rng: np.random.Generator, n: int, alpha: float = 1.7) -> np.ndarray:
    if n == 0:
        return np.zeros(0)
    return rng.pareto(alpha, size=n) + 1.0


def _sample(rng: np.random.Generator, n: int, size: int, weights: np.ndarray | None = None) -> np.ndarray:
    if n == 0 or size == 0:
        return np.zeros(0, dtype=np.int64)
    if weights is None:
        return rng.integers(0, n, size=size)
    p = weights / weights.sum()
    return rng.choice(n, size=size, replace=True, p=p)


def _make_sids(rng: np.random.Generator, n_domains: int, per_domain: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return (sids, domain_idx) such that sids share domain prefixes."""
    domain_prefixes = []
    for _ in range(n_domains):
        a = int(rng.integers(1_000_000_000, 4_000_000_000))
        b = int(rng.integers(1_000_000_000, 4_000_000_000))
        c = int(rng.integers(100_000_000, 999_999_999))
        domain_prefixes.append(f"S-1-5-21-{a}-{b}-{c}")
    sids: list[str] = []
    domain_idx: list[int] = []
    next_rid = [1000] * n_domains
    for d, count in enumerate(per_domain):
        for _ in range(count):
            sids.append(f"{domain_prefixes[d]}-{next_rid[d]}")
            next_rid[d] += 1
            domain_idx.append(d)
    return np.array(sids), np.array(domain_idx, dtype=np.int64), np.array(domain_prefixes)


def generate(scale: ScaleParams, out_dir: Path) -> None:
    rng = np.random.default_rng(scale.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    collected_at = np.int64(1_747_526_400)  # 2026-05-18T00:00:00Z

    # ------------------------------------------------------------------
    # AD half.
    # ------------------------------------------------------------------
    domain_names = np.array([f"corp{i:02d}.example.com".upper() for i in range(scale.n_domains)])
    # Allocate users/groups/computers across domains via power-law per-domain weights.
    dom_user_w = _power_law_weights(rng, scale.n_domains, alpha=1.5)
    dom_grp_w = _power_law_weights(rng, scale.n_domains, alpha=1.5)
    dom_comp_w = _power_law_weights(rng, scale.n_domains, alpha=1.5)

    def _split(total: int, weights: np.ndarray) -> list[int]:
        p = weights / weights.sum()
        counts = np.floor(p * total).astype(np.int64)
        deficit = total - int(counts.sum())
        if deficit > 0:
            top = np.argsort(p)[-deficit:]
            counts[top] += 1
        return counts.tolist()

    users_per_domain = _split(scale.n_ad_users, dom_user_w)
    groups_per_domain = _split(scale.n_ad_groups, dom_grp_w)
    computers_per_domain = _split(scale.n_ad_computers, dom_comp_w)

    user_sids, user_dom_idx, dom_sid_prefix = _make_sids(rng, scale.n_domains, users_per_domain)
    group_sids, group_dom_idx, _ = _make_sids(rng, scale.n_domains, groups_per_domain)
    computer_sids, computer_dom_idx, _ = _make_sids(rng, scale.n_domains, computers_per_domain)

    # AD users.
    user_sam = np.array([f"user{i:06d}" for i in range(scale.n_ad_users)])
    user_display = np.array([f"User {i:06d}" for i in range(scale.n_ad_users)])
    user_enabled = rng.random(scale.n_ad_users) < 0.92
    user_admincount = rng.random(scale.n_ad_users) < 0.03
    user_pwd_never_expires = rng.random(scale.n_ad_users) < 0.10
    user_dont_req_preauth = rng.random(scale.n_ad_users) < 0.01  # AS-REP-roastable
    user_has_spn = rng.random(scale.n_ad_users) < 0.04  # Kerberoastable
    user_sensitive = rng.random(scale.n_ad_users) < 0.02
    user_unconstrained_deleg = rng.random(scale.n_ad_users) < 0.005
    user_pwd_last_set = collected_at - rng.integers(0, 3 * 365 * 86400, size=scale.n_ad_users, dtype=np.int64)
    user_last_logon = collected_at - rng.integers(0, 180 * 86400, size=scale.n_ad_users, dtype=np.int64)

    ad_users_tbl = pa.table(
        {
            "object_id": user_sids,
            "samaccountname": user_sam,
            "displayname": user_display,
            "domain": np.take(domain_names, user_dom_idx),
            "enabled": user_enabled,
            "admincount": user_admincount,
            "pwdneverexpires": user_pwd_never_expires,
            "dontreqpreauth": user_dont_req_preauth,
            "hasspn": user_has_spn,
            "sensitive": user_sensitive,
            "unconstraineddelegation": user_unconstrained_deleg,
            "pwdlastset": user_pwd_last_set,
            "lastlogon": user_last_logon,
            "collected_at": np.full(scale.n_ad_users, collected_at, dtype=np.int64),
        }
    )
    pq.write_table(ad_users_tbl, out_dir / "ad_users.parquet")

    # AD groups.
    group_name = np.array([f"GROUP{i:05d}" for i in range(scale.n_ad_groups)])
    group_admincount = rng.random(scale.n_ad_groups) < 0.08
    group_high_value = rng.random(scale.n_ad_groups) < 0.02  # Domain Admins-like
    ad_groups_tbl = pa.table(
        {
            "object_id": group_sids,
            "name": group_name,
            "domain": np.take(domain_names, group_dom_idx),
            "admincount": group_admincount,
            "high_value": group_high_value,
            "collected_at": np.full(scale.n_ad_groups, collected_at, dtype=np.int64),
        }
    )
    pq.write_table(ad_groups_tbl, out_dir / "ad_groups.parquet")

    # AD computers.
    comp_name = np.array([f"COMP{i:06d}" for i in range(scale.n_ad_computers)])
    comp_os = rng.choice(
        np.array(["Windows Server 2022", "Windows Server 2019", "Windows Server 2016", "Windows 11", "Windows 10"]),
        size=scale.n_ad_computers,
        p=np.array([0.20, 0.25, 0.10, 0.20, 0.25]),
    )
    comp_haslaps = rng.random(scale.n_ad_computers) < 0.55
    comp_unconstrained = rng.random(scale.n_ad_computers) < 0.02
    comp_is_dc = rng.random(scale.n_ad_computers) < 0.01
    comp_enabled = rng.random(scale.n_ad_computers) < 0.95
    ad_computers_tbl = pa.table(
        {
            "object_id": computer_sids,
            "name": comp_name,
            "domain": np.take(domain_names, computer_dom_idx),
            "operating_system": comp_os,
            "haslaps": comp_haslaps,
            "unconstraineddelegation": comp_unconstrained,
            "is_dc": comp_is_dc,
            "enabled": comp_enabled,
            "collected_at": np.full(scale.n_ad_computers, collected_at, dtype=np.int64),
        }
    )
    pq.write_table(ad_computers_tbl, out_dir / "ad_computers.parquet")

    # ACEs. Principal is a user or group; target is any of user/group/computer.
    target_pool = np.concatenate([user_sids, group_sids, computer_sids])
    target_kind = np.concatenate(
        [
            np.full(scale.n_ad_users, "User"),
            np.full(scale.n_ad_groups, "Group"),
            np.full(scale.n_ad_computers, "Computer"),
        ]
    )
    target_weights = _power_law_weights(rng, target_pool.size)
    principal_pool = np.concatenate([user_sids, group_sids])
    principal_kind = np.concatenate(
        [np.full(scale.n_ad_users, "User"), np.full(scale.n_ad_groups, "Group")]
    )
    principal_weights = _power_law_weights(rng, principal_pool.size)

    ace_principal_idx = _sample(rng, principal_pool.size, scale.n_aces, weights=principal_weights)
    ace_target_idx = _sample(rng, target_pool.size, scale.n_aces, weights=target_weights)
    ace_right = rng.choice(ACE_RIGHTS, size=scale.n_aces, p=ACE_RIGHT_WEIGHTS / ACE_RIGHT_WEIGHTS.sum())
    ace_inherited = rng.random(scale.n_aces) < 0.25
    ad_aces_tbl = pa.table(
        {
            "ace_id": np.array([f"ace_{i:08d}" for i in range(scale.n_aces)]),
            "principal_id": np.take(principal_pool, ace_principal_idx),
            "principal_kind": np.take(principal_kind, ace_principal_idx),
            "target_id": np.take(target_pool, ace_target_idx),
            "target_kind": np.take(target_kind, ace_target_idx),
            "right_name": ace_right,
            "is_inherited": ace_inherited,
            "collected_at": np.full(scale.n_aces, collected_at, dtype=np.int64),
        }
    )
    pq.write_table(ad_aces_tbl, out_dir / "ad_aces.parquet")

    # Sessions: user logged on to computer.
    sess_user_w = _power_law_weights(rng, scale.n_ad_users)
    sess_comp_w = _power_law_weights(rng, scale.n_ad_computers)
    sess_user_idx = _sample(rng, scale.n_ad_users, scale.n_sessions, weights=sess_user_w)
    sess_comp_idx = _sample(rng, scale.n_ad_computers, scale.n_sessions, weights=sess_comp_w)
    sess_privileged = rng.random(scale.n_sessions) < 0.08
    ad_sessions_tbl = pa.table(
        {
            "session_id": np.array([f"sess_{i:08d}" for i in range(scale.n_sessions)]),
            "user_id": np.take(user_sids, sess_user_idx),
            "computer_id": np.take(computer_sids, sess_comp_idx),
            "privileged": sess_privileged,
            "collected_at": np.full(scale.n_sessions, collected_at, dtype=np.int64),
        }
    )
    pq.write_table(ad_sessions_tbl, out_dir / "ad_sessions.parquet")

    # Certificate templates (small).
    ct_ids = np.array([f"ct_{i:04d}" for i in range(scale.n_cert_templates)])
    ct_name = np.array([f"Template{i:04d}" for i in range(scale.n_cert_templates)])
    ct_enroll_dangerous = rng.random(scale.n_cert_templates) < 0.30  # ESC1-style
    ct_supplies_subject = rng.random(scale.n_cert_templates) < 0.40
    ct_client_auth = rng.random(scale.n_cert_templates) < 0.70
    ad_cert_tbl = pa.table(
        {
            "template_id": ct_ids,
            "name": ct_name,
            "domain": np.take(domain_names, rng.integers(0, scale.n_domains, size=scale.n_cert_templates)),
            "enrollee_supplies_subject": ct_supplies_subject,
            "client_auth_eku": ct_client_auth,
            "enroll_dangerous": ct_enroll_dangerous,
            "collected_at": np.full(scale.n_cert_templates, collected_at, dtype=np.int64),
        }
    )
    pq.write_table(ad_cert_tbl, out_dir / "ad_cert_templates.parquet")

    # ------------------------------------------------------------------
    # Entra half.
    # ------------------------------------------------------------------
    def _uuid(rng: np.random.Generator) -> str:
        # Deterministic UUID-shaped string from rng bytes.
        b = rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()
        return str(uuid.UUID(bytes=b, version=4))

    tenant_ids = np.array([_uuid(rng) for _ in range(scale.n_tenants)])
    subscription_ids = np.array([_uuid(rng) for _ in range(scale.n_subscriptions)])
    sub_tenant_idx = rng.integers(0, scale.n_tenants, size=scale.n_subscriptions)
    rg_ids = np.array([_uuid(rng) for _ in range(scale.n_resource_groups)])
    rg_sub_idx = rng.integers(0, scale.n_subscriptions, size=scale.n_resource_groups)

    # Azure users.
    az_user_ids = np.array([_uuid(rng) for _ in range(scale.n_az_users)])
    az_user_upn = np.array([f"u{i:06d}@corp.onmicrosoft.com" for i in range(scale.n_az_users)])
    az_user_tenant_idx = rng.integers(0, scale.n_tenants, size=scale.n_az_users)
    az_user_enabled = rng.random(scale.n_az_users) < 0.95
    az_user_mfa = rng.random(scale.n_az_users) < 0.70
    az_user_admin = rng.random(scale.n_az_users) < 0.03

    azure_users_tbl = pa.table(
        {
            "object_id": az_user_ids,
            "upn": az_user_upn,
            "tenant_id": np.take(tenant_ids, az_user_tenant_idx),
            "enabled": az_user_enabled,
            "mfa_enabled": az_user_mfa,
            "is_admin": az_user_admin,
            "collected_at": np.full(scale.n_az_users, collected_at, dtype=np.int64),
        }
    )
    pq.write_table(azure_users_tbl, out_dir / "azure_users.parquet")

    # Azure groups.
    az_group_ids = np.array([_uuid(rng) for _ in range(scale.n_az_groups)])
    az_group_name = np.array([f"AZGroup{i:05d}" for i in range(scale.n_az_groups)])
    az_group_tenant_idx = rng.integers(0, scale.n_tenants, size=scale.n_az_groups)
    az_group_security = rng.random(scale.n_az_groups) < 0.85

    azure_groups_tbl = pa.table(
        {
            "object_id": az_group_ids,
            "name": az_group_name,
            "tenant_id": np.take(tenant_ids, az_group_tenant_idx),
            "security_enabled": az_group_security,
            "collected_at": np.full(scale.n_az_groups, collected_at, dtype=np.int64),
        }
    )
    pq.write_table(azure_groups_tbl, out_dir / "azure_groups.parquet")

    # Azure apps + service principals.
    az_app_ids = np.array([_uuid(rng) for _ in range(scale.n_az_apps)])
    az_app_display = np.array([f"App {i:05d}" for i in range(scale.n_az_apps)])
    az_app_tenant_idx = rng.integers(0, scale.n_tenants, size=scale.n_az_apps)
    az_app_credential_count = rng.integers(0, 4, size=scale.n_az_apps)
    az_app_password_credentials_expired_days = rng.integers(-365, 730, size=scale.n_az_apps)

    azure_apps_tbl = pa.table(
        {
            "object_id": az_app_ids,
            "display_name": az_app_display,
            "tenant_id": np.take(tenant_ids, az_app_tenant_idx),
            "credential_count": az_app_credential_count.astype(np.int64),
            "password_credential_age_days": az_app_password_credentials_expired_days.astype(np.int64),
            "collected_at": np.full(scale.n_az_apps, collected_at, dtype=np.int64),
        }
    )
    pq.write_table(azure_apps_tbl, out_dir / "azure_apps.parquet")

    az_sp_ids = np.array([_uuid(rng) for _ in range(scale.n_az_service_principals)])
    # Each SP belongs to one app (cycle if SP > app, but here equal).
    sp_app_idx = rng.integers(0, scale.n_az_apps, size=scale.n_az_service_principals)
    az_sp_tenant_idx = np.take(az_app_tenant_idx, sp_app_idx)
    az_sp_account_enabled = rng.random(scale.n_az_service_principals) < 0.95
    az_sp_is_privileged = rng.random(scale.n_az_service_principals) < 0.05

    azure_sps_tbl = pa.table(
        {
            "object_id": az_sp_ids,
            "app_id": np.take(az_app_ids, sp_app_idx),
            "tenant_id": np.take(tenant_ids, az_sp_tenant_idx),
            "account_enabled": az_sp_account_enabled,
            "is_privileged": az_sp_is_privileged,
            "collected_at": np.full(scale.n_az_service_principals, collected_at, dtype=np.int64),
        }
    )
    pq.write_table(azure_sps_tbl, out_dir / "azure_service_principals.parquet")

    # Azure VMs (resources).
    az_vm_ids = np.array([_uuid(rng) for _ in range(scale.n_az_vms)])
    az_vm_name = np.array([f"vm-{i:05d}" for i in range(scale.n_az_vms)])
    vm_rg_idx = rng.integers(0, scale.n_resource_groups, size=scale.n_az_vms)
    vm_sub_idx = np.take(rg_sub_idx, vm_rg_idx)

    # Azure role assignments.
    # Principals: 70% user, 20% service principal, 10% group.
    n_ra = scale.n_az_role_assignments
    ra_principal_kind = rng.choice(
        np.array(["user", "service_principal", "group"]),
        size=n_ra,
        p=np.array([0.70, 0.20, 0.10]),
    )
    ra_user_w = _power_law_weights(rng, scale.n_az_users)
    ra_sp_w = _power_law_weights(rng, scale.n_az_service_principals)
    ra_grp_w = _power_law_weights(rng, scale.n_az_groups)
    ra_user_idx = _sample(rng, scale.n_az_users, n_ra, weights=ra_user_w)
    ra_sp_idx = _sample(rng, scale.n_az_service_principals, n_ra, weights=ra_sp_w)
    ra_grp_idx = _sample(rng, scale.n_az_groups, n_ra, weights=ra_grp_w)
    ra_principal_id = np.where(
        ra_principal_kind == "user",
        np.take(az_user_ids, ra_user_idx),
        np.where(
            ra_principal_kind == "service_principal",
            np.take(az_sp_ids, ra_sp_idx),
            np.take(az_group_ids, ra_grp_idx),
        ),
    )

    # Scope target: 70% VM, 20% RG, 10% subscription.
    ra_scope_kind = rng.choice(
        np.array(["vm", "resource_group", "subscription"]),
        size=n_ra,
        p=np.array([0.70, 0.20, 0.10]),
    )
    ra_vm_w = _power_law_weights(rng, scale.n_az_vms)
    ra_vm_idx = _sample(rng, scale.n_az_vms, n_ra, weights=ra_vm_w)
    ra_rg_idx = rng.integers(0, scale.n_resource_groups, size=n_ra)
    ra_sub_idx = rng.integers(0, scale.n_subscriptions, size=n_ra)
    ra_scope_id = np.where(
        ra_scope_kind == "vm",
        np.take(az_vm_ids, ra_vm_idx),
        np.where(
            ra_scope_kind == "resource_group",
            np.take(rg_ids, ra_rg_idx),
            np.take(subscription_ids, ra_sub_idx),
        ),
    )

    ra_role_name = rng.choice(AZ_BUILTIN_ROLES, size=n_ra, p=AZ_ROLE_WEIGHTS / AZ_ROLE_WEIGHTS.sum())

    azure_ra_tbl = pa.table(
        {
            "role_assignment_id": np.array([_uuid(rng) for _ in range(n_ra)]),
            "principal_id": ra_principal_id,
            "principal_kind": ra_principal_kind,
            "scope_id": ra_scope_id,
            "scope_kind": ra_scope_kind,
            "role_name": ra_role_name,
            "collected_at": np.full(n_ra, collected_at, dtype=np.int64),
        }
    )
    pq.write_table(azure_ra_tbl, out_dir / "azure_role_assignments.parquet")

    # ------------------------------------------------------------------
    # Hybrid sync: AD user <-> AZ user.
    # ------------------------------------------------------------------
    n_sync = min(int(round(scale.n_ad_users * scale.ad_sync_fraction)), scale.n_az_users)
    ad_sync_pick = rng.choice(scale.n_ad_users, size=n_sync, replace=False)
    az_sync_pick = rng.choice(scale.n_az_users, size=n_sync, replace=False)
    sync_ad_sid = np.take(user_sids, ad_sync_pick)
    sync_az_id = np.take(az_user_ids, az_sync_pick)

    # ------------------------------------------------------------------
    # bh_nodes / bh_edges (AD + Entra slice).
    # ------------------------------------------------------------------
    node_id_chunks: list[np.ndarray] = []
    node_kind_chunks: list[np.ndarray] = []
    node_name_chunks: list[np.ndarray] = []
    node_env_chunks: list[np.ndarray] = []
    node_tenant_chunks: list[np.ndarray] = []
    node_source_chunks: list[np.ndarray] = []
    node_high_value_chunks: list[np.ndarray] = []

    def _push_nodes(
        ids: np.ndarray,
        kind: str,
        names: np.ndarray,
        env_ids: np.ndarray,
        tenant_ids_for_nodes: np.ndarray,
        source_kind: str,
        high_value: np.ndarray | None = None,
    ) -> None:
        node_id_chunks.append(ids)
        node_kind_chunks.append(np.full(ids.size, kind))
        node_name_chunks.append(names)
        node_env_chunks.append(env_ids)
        node_tenant_chunks.append(tenant_ids_for_nodes)
        node_source_chunks.append(np.full(ids.size, source_kind))
        node_high_value_chunks.append(
            high_value if high_value is not None else np.zeros(ids.size, dtype=bool)
        )

    # AD nodes.
    _push_nodes(user_sids, "ADUser", user_sam,
                np.take(domain_names, user_dom_idx),
                np.take(domain_names, user_dom_idx),
                "active_directory",
                user_admincount)
    _push_nodes(group_sids, "ADGroup", group_name,
                np.take(domain_names, group_dom_idx),
                np.take(domain_names, group_dom_idx),
                "active_directory",
                group_high_value)
    _push_nodes(computer_sids, "ADComputer", comp_name,
                np.take(domain_names, computer_dom_idx),
                np.take(domain_names, computer_dom_idx),
                "active_directory",
                comp_is_dc)
    # Domain nodes themselves (objectid = SID prefix).
    _push_nodes(dom_sid_prefix, "ADDomain", domain_names, domain_names, domain_names,
                "active_directory", np.ones(scale.n_domains, dtype=bool))
    _push_nodes(ct_ids, "ADCertTemplate", ct_name,
                np.take(domain_names, rng.integers(0, scale.n_domains, size=scale.n_cert_templates)),
                np.take(domain_names, rng.integers(0, scale.n_domains, size=scale.n_cert_templates)),
                "active_directory")

    # Entra nodes.
    _push_nodes(tenant_ids, "AZTenant", tenant_ids, tenant_ids, tenant_ids,
                "azure", np.ones(scale.n_tenants, dtype=bool))
    _push_nodes(subscription_ids, "AZSubscription", subscription_ids,
                np.take(tenant_ids, sub_tenant_idx),
                np.take(tenant_ids, sub_tenant_idx),
                "azure")
    _push_nodes(rg_ids, "AZResourceGroup", rg_ids,
                np.take(tenant_ids, np.take(sub_tenant_idx, rg_sub_idx)),
                np.take(tenant_ids, np.take(sub_tenant_idx, rg_sub_idx)),
                "azure")
    _push_nodes(az_user_ids, "AZUser", az_user_upn,
                np.take(tenant_ids, az_user_tenant_idx),
                np.take(tenant_ids, az_user_tenant_idx),
                "azure",
                az_user_admin)
    _push_nodes(az_group_ids, "AZGroup", az_group_name,
                np.take(tenant_ids, az_group_tenant_idx),
                np.take(tenant_ids, az_group_tenant_idx),
                "azure")
    _push_nodes(az_app_ids, "AZApp", az_app_display,
                np.take(tenant_ids, az_app_tenant_idx),
                np.take(tenant_ids, az_app_tenant_idx),
                "azure")
    _push_nodes(az_sp_ids, "AZServicePrincipal", az_sp_ids,
                np.take(tenant_ids, az_sp_tenant_idx),
                np.take(tenant_ids, az_sp_tenant_idx),
                "azure",
                az_sp_is_privileged)
    _push_nodes(az_vm_ids, "AZVM", az_vm_name,
                np.take(tenant_ids, np.take(sub_tenant_idx, vm_sub_idx)),
                np.take(tenant_ids, np.take(sub_tenant_idx, vm_sub_idx)),
                "azure")

    node_ids = np.concatenate(node_id_chunks)
    n_nodes = node_ids.size
    nodes_tbl = pa.table(
        {
            "node_id": node_ids,
            "primary_kind": np.concatenate(node_kind_chunks),
            "source_kind": np.concatenate(node_source_chunks),
            "environment_id": np.concatenate(node_env_chunks),
            "tenant_id": np.concatenate(node_tenant_chunks),
            "name": np.concatenate(node_name_chunks),
            "displayname": np.concatenate(node_name_chunks),
            "risk_tier": np.zeros(n_nodes, dtype=np.int32),
            "high_value": np.concatenate(node_high_value_chunks),
            "collected": np.ones(n_nodes, dtype=bool),
            "collected_at": np.full(n_nodes, collected_at, dtype=np.int64),
        }
    )
    pq.write_table(nodes_tbl, out_dir / "bh_nodes.parquet")

    # Edges.
    edge_src: list[np.ndarray] = []
    edge_dst: list[np.ndarray] = []
    edge_kind: list[np.ndarray] = []
    edge_trav: list[np.ndarray] = []
    edge_src_kind: list[np.ndarray] = []
    edge_env: list[np.ndarray] = []
    edge_tenant: list[np.ndarray] = []

    def _push_edges(
        src: np.ndarray,
        dst: np.ndarray,
        kind: str,
        traversable: bool,
        source_kind: str,
        env: np.ndarray,
        tenant: np.ndarray,
    ) -> None:
        if src.size == 0:
            return
        edge_src.append(src)
        edge_dst.append(dst)
        edge_kind.append(np.full(src.size, kind))
        edge_trav.append(np.full(src.size, traversable))
        edge_src_kind.append(np.full(src.size, source_kind))
        edge_env.append(env)
        edge_tenant.append(tenant)

    # ADContains: domain -> user/group/computer.
    _push_edges(
        np.take(dom_sid_prefix, user_dom_idx),
        user_sids,
        "ADContains",
        True,
        "active_directory",
        np.take(domain_names, user_dom_idx),
        np.take(domain_names, user_dom_idx),
    )
    _push_edges(
        np.take(dom_sid_prefix, group_dom_idx),
        group_sids,
        "ADContains",
        True,
        "active_directory",
        np.take(domain_names, group_dom_idx),
        np.take(domain_names, group_dom_idx),
    )
    _push_edges(
        np.take(dom_sid_prefix, computer_dom_idx),
        computer_sids,
        "ADContains",
        True,
        "active_directory",
        np.take(domain_names, computer_dom_idx),
        np.take(domain_names, computer_dom_idx),
    )

    # ADMemberOf: assign each user to 1-5 groups in the same domain; nested groups too.
    user_n_groups = rng.integers(1, 6, size=scale.n_ad_users)
    membership_user_idx = np.repeat(np.arange(scale.n_ad_users), user_n_groups)
    n_user_mem = membership_user_idx.size
    membership_group_idx = np.zeros(n_user_mem, dtype=np.int64)
    # For each membership row pick a group in same domain. Build per-domain group index lists.
    groups_in_domain = [np.where(group_dom_idx == d)[0] for d in range(scale.n_domains)]
    group_weights_in_domain = [_power_law_weights(rng, len(g)) for g in groups_in_domain]
    for i in range(n_user_mem):
        d = int(user_dom_idx[membership_user_idx[i]])
        pool = groups_in_domain[d]
        if pool.size == 0:
            membership_group_idx[i] = int(rng.integers(0, scale.n_ad_groups))
        else:
            w = group_weights_in_domain[d]
            membership_group_idx[i] = int(rng.choice(pool, p=w / w.sum()))
    _push_edges(
        np.take(user_sids, membership_user_idx),
        np.take(group_sids, membership_group_idx),
        "ADMemberOf",
        True,
        "active_directory",
        np.take(domain_names, user_dom_idx[membership_user_idx]),
        np.take(domain_names, user_dom_idx[membership_user_idx]),
    )

    # Nested groups: ~30% of groups are nested under another group in same domain (lower index).
    nested_mask = rng.random(scale.n_ad_groups) < 0.30
    nested_idx = np.where(nested_mask)[0]
    nested_parent = np.full(nested_idx.size, -1, dtype=np.int64)
    for i, gi in enumerate(nested_idx):
        d = int(group_dom_idx[gi])
        candidates = np.where((group_dom_idx[:gi] == d))[0]
        if candidates.size > 0:
            nested_parent[i] = int(rng.choice(candidates))
    has_parent = nested_parent >= 0
    _push_edges(
        np.take(group_sids, nested_idx[has_parent]),
        np.take(group_sids, nested_parent[has_parent]),
        "ADMemberOf",
        True,
        "active_directory",
        np.take(domain_names, group_dom_idx[nested_idx[has_parent]]),
        np.take(domain_names, group_dom_idx[nested_idx[has_parent]]),
    )

    # ADHasSession: computer -> user.
    _push_edges(
        np.take(computer_sids, sess_comp_idx),
        np.take(user_sids, sess_user_idx),
        "ADHasSession",
        True,
        "active_directory",
        np.take(domain_names, computer_dom_idx[sess_comp_idx]),
        np.take(domain_names, computer_dom_idx[sess_comp_idx]),
    )

    # ACE-derived edges, one kind per right.
    for right in ACE_RIGHTS:
        mask = ace_right == right
        if not mask.any():
            continue
        # Use the principal's domain as env/tenant.
        principal_id = np.take(principal_pool, ace_principal_idx[mask])
        # Map principal back to a domain by looking up domain idx via concatenation of user/group dom idx.
        principal_dom_lookup = np.concatenate([user_dom_idx, group_dom_idx])
        principal_dom_idx = np.take(principal_dom_lookup, ace_principal_idx[mask])
        _push_edges(
            principal_id,
            np.take(target_pool, ace_target_idx[mask]),
            f"AD{right}",
            True,
            "active_directory",
            np.take(domain_names, principal_dom_idx),
            np.take(domain_names, principal_dom_idx),
        )

    # AZContains: tenant -> subscription, subscription -> RG, RG -> VM, tenant -> users/groups/apps/SPs.
    _push_edges(
        np.take(tenant_ids, sub_tenant_idx),
        subscription_ids,
        "AZContains",
        True,
        "azure",
        np.take(tenant_ids, sub_tenant_idx),
        np.take(tenant_ids, sub_tenant_idx),
    )
    _push_edges(
        np.take(subscription_ids, rg_sub_idx),
        rg_ids,
        "AZContains",
        True,
        "azure",
        np.take(tenant_ids, np.take(sub_tenant_idx, rg_sub_idx)),
        np.take(tenant_ids, np.take(sub_tenant_idx, rg_sub_idx)),
    )
    _push_edges(
        np.take(rg_ids, vm_rg_idx),
        az_vm_ids,
        "AZContains",
        True,
        "azure",
        np.take(tenant_ids, np.take(sub_tenant_idx, vm_sub_idx)),
        np.take(tenant_ids, np.take(sub_tenant_idx, vm_sub_idx)),
    )

    # AZ App -> ServicePrincipal (owns / runs as).
    _push_edges(
        np.take(az_app_ids, sp_app_idx),
        az_sp_ids,
        "AZRunsAs",
        True,
        "azure",
        np.take(tenant_ids, az_sp_tenant_idx),
        np.take(tenant_ids, az_sp_tenant_idx),
    )

    # AZ group membership: 1-3 groups per user (same tenant).
    az_user_n_groups = rng.integers(1, 4, size=scale.n_az_users)
    azm_user_idx = np.repeat(np.arange(scale.n_az_users), az_user_n_groups)
    n_azm = azm_user_idx.size
    azm_group_idx = np.zeros(n_azm, dtype=np.int64)
    az_groups_in_tenant = [np.where(az_group_tenant_idx == t)[0] for t in range(scale.n_tenants)]
    az_group_w_in_tenant = [_power_law_weights(rng, len(g)) for g in az_groups_in_tenant]
    for i in range(n_azm):
        t = int(az_user_tenant_idx[azm_user_idx[i]])
        pool = az_groups_in_tenant[t]
        if pool.size == 0:
            azm_group_idx[i] = int(rng.integers(0, scale.n_az_groups))
        else:
            w = az_group_w_in_tenant[t]
            azm_group_idx[i] = int(rng.choice(pool, p=w / w.sum()))
    _push_edges(
        np.take(az_user_ids, azm_user_idx),
        np.take(az_group_ids, azm_group_idx),
        "AZMemberOf",
        True,
        "azure",
        np.take(tenant_ids, az_user_tenant_idx[azm_user_idx]),
        np.take(tenant_ids, az_user_tenant_idx[azm_user_idx]),
    )

    # AZ role assignments -> AZHasRole_<roleName>.
    for role in AZ_BUILTIN_ROLES:
        mask = ra_role_name == role
        if not mask.any():
            continue
        kind_label = role.replace(" ", "")
        # tenant lookup follows principal kind.
        prin_tenant = np.where(
            ra_principal_kind[mask] == "user",
            np.take(tenant_ids, az_user_tenant_idx[ra_user_idx[mask]]),
            np.where(
                ra_principal_kind[mask] == "service_principal",
                np.take(tenant_ids, az_sp_tenant_idx[ra_sp_idx[mask]]),
                np.take(tenant_ids, az_group_tenant_idx[ra_grp_idx[mask]]),
            ),
        )
        _push_edges(
            ra_principal_id[mask],
            ra_scope_id[mask],
            f"AZHasRole_{kind_label}",
            True,
            "azure",
            prin_tenant,
            prin_tenant,
        )

    # Hybrid bridge: AD user -> AZ user.
    if n_sync > 0:
        _push_edges(
            sync_ad_sid,
            sync_az_id,
            "ADSyncedTo",
            True,
            "hybrid",
            np.take(domain_names, user_dom_idx[ad_sync_pick]),
            np.take(tenant_ids, az_user_tenant_idx[az_sync_pick]),
        )

    src = np.concatenate(edge_src)
    dst = np.concatenate(edge_dst)
    n_edges = src.size
    edges_tbl = pa.table(
        {
            "edge_id": np.array([f"e_{i:09d}" for i in range(n_edges)]),
            "src_id": src,
            "dst_id": dst,
            "edge_kind": np.concatenate(edge_kind),
            "traversable": np.concatenate(edge_trav),
            "source_kind": np.concatenate(edge_src_kind),
            "environment_id": np.concatenate(edge_env),
            "tenant_id": np.concatenate(edge_tenant),
            "collected_at": np.full(n_edges, collected_at, dtype=np.int64),
            "weight": np.ones(n_edges, dtype=np.float64),
        }
    )
    pq.write_table(edges_tbl, out_dir / "bh_edges.parquet")

    print(
        f"[gen_ad_entra_bh] scale={scale.name} "
        f"ad_domains={scale.n_domains} ad_users={scale.n_ad_users} "
        f"ad_groups={scale.n_ad_groups} ad_computers={scale.n_ad_computers} "
        f"ad_aces={scale.n_aces} ad_sessions={scale.n_sessions} "
        f"ad_cert_templates={scale.n_cert_templates} "
        f"az_tenants={scale.n_tenants} az_subs={scale.n_subscriptions} "
        f"az_rgs={scale.n_resource_groups} az_users={scale.n_az_users} "
        f"az_groups={scale.n_az_groups} az_apps={scale.n_az_apps} "
        f"az_sps={scale.n_az_service_principals} az_vms={scale.n_az_vms} "
        f"az_ras={scale.n_az_role_assignments} "
        f"hybrid_sync={n_sync} "
        f"bh_nodes={n_nodes} bh_edges={n_edges}",
        file=sys.stderr,
    )
    print(f"[gen_ad_entra_bh] wrote {out_dir}", file=sys.stderr)


def main() -> int:
    scale_name = os.environ.get("BH_ADENTRA_SCALE", "bh_sf1")
    if scale_name not in SCALE_PRESETS:
        print(
            f"[gen_ad_entra_bh] unsupported scale: {scale_name!r}; available: "
            f"{sorted(SCALE_PRESETS)}",
            file=sys.stderr,
        )
        return 2
    scale = SCALE_PRESETS[scale_name]
    seed_override = os.environ.get("BH_ADENTRA_SEED")
    if seed_override:
        scale = ScaleParams(**{**scale.__dict__, "seed": int(seed_override)})

    default_out = REPO_ROOT / "fixture" / "data" / "ad_entra" / scale.name / "parquet"
    out_dir = Path(os.environ.get("BH_ADENTRA_OUT_DIR", str(default_out))).resolve()
    generate(scale, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
