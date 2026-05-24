"""Generate a synthetic GitHub OpenGraph BloodHound fixture as Parquet.

Implements the GitHub half of the first slice described in
`docs/e2e/2026-05-19-bloodhound-graph-fixture-plan.md`. Emits one set of
typed wide tables plus a GitHub-scoped slice of the generic `bh_nodes` /
`bh_edges` layer.

Tables emitted:
  github_repositories
  github_branch_rules
  github_workflows
  github_secrets
  github_role_assignments
  bh_nodes        (GitHub-scoped slice)
  bh_edges        (GitHub-scoped slice)

Schema coverage is intentionally a subset of the 26 node / 114 edge GitHub
OpenGraph catalogue: organization, repository, branch, branch protection rule,
workflow, secret, user, team, role.

Env:
  BH_GITHUB_SCALE      bh_sf1 (default) | bh_sf10 | bh_sf100
  BH_GITHUB_OUT_DIR    output root; default <repo>/fixture/data/github/<scale>/parquet
  BH_GITHUB_SEED       override deterministic seed (default derived from scale)

`bh_sf1` target shape (per design note §First-slice generator decisions):
  10 orgs, ~500 repos, 2,000 users, 200 teams, ~1,500 branch rules,
  ~2,000 workflows, ~3,000 secrets, ~50k repo role assignments,
  total edges in the 100k-300k range.

Run:
  python fixture/scripts/gen_github_bh.py
  BH_GITHUB_SCALE=bh_sf1 python fixture/scripts/gen_github_bh.py
"""

import os
import sys
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
    n_orgs: int
    repos_per_org_mean: float
    n_users: int
    n_teams: int
    branch_rules_total: int
    workflows_total: int
    secrets_total: int
    role_assignments_total: int


SCALE_PRESETS: dict[str, ScaleParams] = {
    "bh_sf1": ScaleParams(
        name="bh_sf1",
        seed=20260518,
        n_orgs=10,
        repos_per_org_mean=50.0,
        n_users=2_000,
        n_teams=200,
        branch_rules_total=1_500,
        workflows_total=2_000,
        secrets_total=3_000,
        role_assignments_total=50_000,
    ),
    # sf10 / sf100 presets are intentionally omitted until the sf1 pass is
    # validated end-to-end. Add them here after the first parity smoke.
}

REPO_ROLES = ("admin", "maintain", "write", "triage", "read")
REPO_ROLE_WEIGHTS = np.array([0.05, 0.10, 0.25, 0.20, 0.40])
SECRET_SCOPES = ("organization", "repository", "environment")
SECRET_SCOPE_WEIGHTS = np.array([0.15, 0.65, 0.20])
BRANCH_NAMES = ("main", "release", "develop", "staging", "hotfix")


def _power_law_split(rng: np.random.Generator, total: int, buckets: int, alpha: float = 1.7) -> np.ndarray:
    """Distribute `total` across `buckets` with a power-law skew. Returns ints summing to `total`."""
    if buckets <= 0:
        return np.zeros(0, dtype=np.int64)
    raw = rng.pareto(alpha, size=buckets) + 1.0
    raw = raw / raw.sum()
    counts = np.floor(raw * total).astype(np.int64)
    deficit = total - counts.sum()
    if deficit > 0:
        top = np.argsort(raw)[-deficit:]
        counts[top] += 1
    return counts


def _sample_indices(rng: np.random.Generator, n: int, size: int, weights: np.ndarray | None = None) -> np.ndarray:
    if n == 0 or size == 0:
        return np.zeros(0, dtype=np.int64)
    if weights is None:
        return rng.integers(0, n, size=size)
    p = weights / weights.sum()
    return rng.choice(n, size=size, replace=True, p=p)


def _power_law_weights(rng: np.random.Generator, n: int, alpha: float = 1.7) -> np.ndarray:
    if n == 0:
        return np.zeros(0)
    raw = rng.pareto(alpha, size=n) + 1.0
    return raw


def generate(scale: ScaleParams, out_dir: Path) -> None:
    rng = np.random.default_rng(scale.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    collected_at = np.int64(1_747_526_400)  # 2026-05-18T00:00:00Z

    # Organizations.
    org_ids = np.array([f"org_{i:04d}" for i in range(scale.n_orgs)])
    org_names = np.array([f"org-{i:04d}" for i in range(scale.n_orgs)])

    # Repos per org via power-law.
    n_repos_total = int(round(scale.repos_per_org_mean * scale.n_orgs))
    repos_per_org = _power_law_split(rng, n_repos_total, scale.n_orgs)
    repo_org_idx = np.repeat(np.arange(scale.n_orgs), repos_per_org)
    n_repos_total = int(repo_org_idx.size)
    repo_ids = np.array([f"repo_{i:06d}" for i in range(n_repos_total)])
    repo_names = np.array(
        [f"{org_names[repo_org_idx[i]]}/repo-{i:06d}" for i in range(n_repos_total)]
    )
    repo_visibility = rng.choice(
        np.array(["public", "private", "internal"]),
        size=n_repos_total,
        p=np.array([0.20, 0.70, 0.10]),
    )
    repo_archived = rng.random(n_repos_total) < 0.05
    repo_default_branch = rng.choice(
        np.array(["main", "master", "develop"]), size=n_repos_total, p=np.array([0.85, 0.10, 0.05])
    )
    repo_stars = rng.integers(0, 5_000, size=n_repos_total)
    repo_created_at = collected_at - rng.integers(0, 5 * 365 * 86400, size=n_repos_total, dtype=np.int64)

    # Users (global, then assigned to orgs by power-law weighting).
    user_ids = np.array([f"user_{i:06d}" for i in range(scale.n_users)])
    user_logins = np.array([f"u{i:06d}" for i in range(scale.n_users)])
    user_mfa = rng.random(scale.n_users) < 0.75
    user_site_admin = rng.random(scale.n_users) < 0.01

    # Teams.
    team_ids = np.array([f"team_{i:05d}" for i in range(scale.n_teams)])
    team_org_idx = rng.integers(0, scale.n_orgs, size=scale.n_teams)
    team_parent_idx = np.full(scale.n_teams, -1, dtype=np.int64)
    for i in range(scale.n_teams):
        # ~30% of teams have a parent within the same org and lower index.
        if i > 0 and rng.random() < 0.30:
            same_org = np.where(team_org_idx[:i] == team_org_idx[i])[0]
            if same_org.size > 0:
                team_parent_idx[i] = int(rng.choice(same_org))
    team_names = np.array(
        [f"{org_names[team_org_idx[i]]}/team-{i:05d}" for i in range(scale.n_teams)]
    )

    # Repository-level branch protection rules.
    rule_repo_idx = _sample_indices(
        rng, n_repos_total, scale.branch_rules_total, weights=_power_law_weights(rng, n_repos_total)
    )
    rule_branch = rng.choice(BRANCH_NAMES, size=scale.branch_rules_total)
    rule_ids = np.array([f"rule_{i:07d}" for i in range(scale.branch_rules_total)])
    rule_requires_review = rng.random(scale.branch_rules_total) < 0.70
    rule_required_reviewers = np.where(
        rule_requires_review, rng.integers(1, 4, size=scale.branch_rules_total), 0
    )
    rule_enforce_admins = rng.random(scale.branch_rules_total) < 0.45
    rule_allow_force_push = rng.random(scale.branch_rules_total) < 0.15
    rule_allow_deletions = rng.random(scale.branch_rules_total) < 0.10

    # Workflows.
    workflow_repo_idx = _sample_indices(
        rng, n_repos_total, scale.workflows_total, weights=_power_law_weights(rng, n_repos_total)
    )
    workflow_ids = np.array([f"wf_{i:07d}" for i in range(scale.workflows_total)])
    workflow_names = np.array(
        [f"workflow-{i:07d}.yml" for i in range(scale.workflows_total)]
    )
    workflow_state = rng.choice(
        np.array(["active", "disabled_manually", "disabled_inactivity"]),
        size=scale.workflows_total,
        p=np.array([0.85, 0.10, 0.05]),
    )
    workflow_has_oidc = rng.random(scale.workflows_total) < 0.30

    # Secrets.
    secret_scope = rng.choice(SECRET_SCOPES, size=scale.secrets_total, p=SECRET_SCOPE_WEIGHTS)
    secret_ids = np.array([f"secret_{i:07d}" for i in range(scale.secrets_total)])
    secret_names = np.array([f"SECRET_{i:07d}" for i in range(scale.secrets_total)])
    secret_repo_idx = np.full(scale.secrets_total, -1, dtype=np.int64)
    secret_org_idx = np.full(scale.secrets_total, -1, dtype=np.int64)
    repo_weights = _power_law_weights(rng, n_repos_total)
    for i in range(scale.secrets_total):
        if secret_scope[i] == "organization":
            secret_org_idx[i] = int(rng.integers(0, scale.n_orgs))
        else:
            secret_repo_idx[i] = int(_sample_indices(rng, n_repos_total, 1, weights=repo_weights)[0])
    secret_leaked = rng.random(scale.secrets_total) < 0.04
    secret_age_days = rng.integers(0, 720, size=scale.secrets_total)

    # Repo role assignments. Mix users and teams as principals.
    n_role = scale.role_assignments_total
    role_principal_kind = rng.choice(np.array(["user", "team"]), size=n_role, p=np.array([0.70, 0.30]))
    user_weights = _power_law_weights(rng, scale.n_users)
    team_weights = _power_law_weights(rng, scale.n_teams)
    role_user_idx = _sample_indices(rng, scale.n_users, n_role, weights=user_weights)
    role_team_idx = _sample_indices(rng, scale.n_teams, n_role, weights=team_weights)
    role_repo_idx = _sample_indices(rng, n_repos_total, n_role, weights=repo_weights)
    role_kind = rng.choice(REPO_ROLES, size=n_role, p=REPO_ROLE_WEIGHTS / REPO_ROLE_WEIGHTS.sum())
    role_ids = np.array([f"role_{i:08d}" for i in range(n_role)])
    role_principal_id = np.where(
        role_principal_kind == "user",
        np.take(user_ids, role_user_idx),
        np.take(team_ids, role_team_idx),
    )

    # ------------------------------------------------------------------
    # Write typed wide tables.
    # ------------------------------------------------------------------
    repos_tbl = pa.table(
        {
            "repo_id": repo_ids,
            "name": repo_names,
            "org_id": np.take(org_ids, repo_org_idx),
            "visibility": repo_visibility,
            "archived": repo_archived,
            "default_branch": repo_default_branch,
            "stars": repo_stars.astype(np.int64),
            "created_at": repo_created_at,
            "collected_at": np.full(n_repos_total, collected_at, dtype=np.int64),
        }
    )
    pq.write_table(repos_tbl, out_dir / "github_repositories.parquet")

    branch_rules_tbl = pa.table(
        {
            "rule_id": rule_ids,
            "repo_id": np.take(repo_ids, rule_repo_idx),
            "branch": rule_branch,
            "requires_review": rule_requires_review,
            "required_reviewers": rule_required_reviewers.astype(np.int64),
            "enforce_admins": rule_enforce_admins,
            "allow_force_push": rule_allow_force_push,
            "allow_deletions": rule_allow_deletions,
            "collected_at": np.full(scale.branch_rules_total, collected_at, dtype=np.int64),
        }
    )
    pq.write_table(branch_rules_tbl, out_dir / "github_branch_rules.parquet")

    workflows_tbl = pa.table(
        {
            "workflow_id": workflow_ids,
            "repo_id": np.take(repo_ids, workflow_repo_idx),
            "name": workflow_names,
            "state": workflow_state,
            "has_oidc": workflow_has_oidc,
            "collected_at": np.full(scale.workflows_total, collected_at, dtype=np.int64),
        }
    )
    pq.write_table(workflows_tbl, out_dir / "github_workflows.parquet")

    secrets_owner_id = np.where(
        secret_scope == "organization",
        np.take(org_ids, np.where(secret_org_idx >= 0, secret_org_idx, 0)),
        np.take(repo_ids, np.where(secret_repo_idx >= 0, secret_repo_idx, 0)),
    )
    secrets_tbl = pa.table(
        {
            "secret_id": secret_ids,
            "name": secret_names,
            "scope": secret_scope,
            "owner_id": secrets_owner_id,
            "leaked": secret_leaked,
            "age_days": secret_age_days.astype(np.int64),
            "collected_at": np.full(scale.secrets_total, collected_at, dtype=np.int64),
        }
    )
    pq.write_table(secrets_tbl, out_dir / "github_secrets.parquet")

    role_tbl = pa.table(
        {
            "role_assignment_id": role_ids,
            "principal_id": role_principal_id,
            "principal_kind": role_principal_kind,
            "repo_id": np.take(repo_ids, role_repo_idx),
            "role": role_kind,
            "collected_at": np.full(n_role, collected_at, dtype=np.int64),
        }
    )
    pq.write_table(role_tbl, out_dir / "github_role_assignments.parquet")

    # ------------------------------------------------------------------
    # Generic graph layer: bh_nodes / bh_edges (GitHub slice).
    # ------------------------------------------------------------------
    node_id_chunks: list[np.ndarray] = []
    node_kind_chunks: list[np.ndarray] = []
    node_name_chunks: list[np.ndarray] = []
    node_env_chunks: list[np.ndarray] = []

    def _push_nodes(ids: np.ndarray, kind: str, names: np.ndarray, env_ids: np.ndarray) -> None:
        node_id_chunks.append(ids)
        node_kind_chunks.append(np.full(ids.size, kind))
        node_name_chunks.append(names)
        node_env_chunks.append(env_ids)

    _push_nodes(org_ids, "GHOrganization", org_names, org_ids)
    _push_nodes(repo_ids, "GHRepository", repo_names, np.take(org_ids, repo_org_idx))
    _push_nodes(user_ids, "GHUser", user_logins, np.full(scale.n_users, "global"))
    _push_nodes(team_ids, "GHTeam", team_names, np.take(org_ids, team_org_idx))
    _push_nodes(workflow_ids, "GHWorkflow", workflow_names, np.take(org_ids, repo_org_idx[workflow_repo_idx]))
    _push_nodes(secret_ids, "GHSecret", secret_names, secrets_owner_id)
    _push_nodes(rule_ids, "GHBranchProtectionRule", rule_ids, np.take(org_ids, repo_org_idx[rule_repo_idx]))

    node_ids = np.concatenate(node_id_chunks)
    node_kinds = np.concatenate(node_kind_chunks)
    node_names = np.concatenate(node_name_chunks)
    node_envs = np.concatenate(node_env_chunks)
    n_nodes = node_ids.size
    nodes_tbl = pa.table(
        {
            "node_id": node_ids,
            "primary_kind": node_kinds,
            "source_kind": np.full(n_nodes, "github"),
            "environment_id": node_envs,
            "tenant_id": np.full(n_nodes, "github"),
            "name": node_names,
            "displayname": node_names,
            "risk_tier": np.zeros(n_nodes, dtype=np.int32),
            "high_value": np.zeros(n_nodes, dtype=bool),
            "collected": np.ones(n_nodes, dtype=bool),
            "collected_at": np.full(n_nodes, collected_at, dtype=np.int64),
        }
    )
    pq.write_table(nodes_tbl, out_dir / "bh_nodes.parquet")

    # Edges: org->repo (Contains), org->team (HasTeam), team->user (HasMember),
    # team->team (HasChild), user/team->repo via role (HasRole<role>),
    # repo->branch_rule (HasBranchRule), repo->workflow (HasWorkflow),
    # repo/org->secret (HasSecret), workflow->repo (RunsIn).
    edge_src: list[np.ndarray] = []
    edge_dst: list[np.ndarray] = []
    edge_kind: list[np.ndarray] = []
    edge_traversable: list[np.ndarray] = []
    edge_env: list[np.ndarray] = []

    def _push_edges(src: np.ndarray, dst: np.ndarray, kind: str, traversable: bool, env_ids: np.ndarray) -> None:
        if src.size == 0:
            return
        edge_src.append(src)
        edge_dst.append(dst)
        edge_kind.append(np.full(src.size, kind))
        edge_traversable.append(np.full(src.size, traversable))
        edge_env.append(env_ids)

    _push_edges(
        np.take(org_ids, repo_org_idx),
        repo_ids,
        "GHContains",
        True,
        np.take(org_ids, repo_org_idx),
    )
    _push_edges(
        np.take(org_ids, team_org_idx),
        team_ids,
        "GHHasTeam",
        True,
        np.take(org_ids, team_org_idx),
    )

    # team -> parent team
    has_parent = team_parent_idx >= 0
    if has_parent.any():
        _push_edges(
            team_ids[has_parent],
            np.take(team_ids, team_parent_idx[has_parent]),
            "GHHasParentTeam",
            True,
            np.take(org_ids, team_org_idx[has_parent]),
        )

    # role assignments -> HasRole<role>
    for role in REPO_ROLES:
        mask = role_kind == role
        if mask.any():
            _push_edges(
                role_principal_id[mask],
                np.take(repo_ids, role_repo_idx[mask]),
                f"GHHasRole_{role}",
                True,
                np.take(org_ids, repo_org_idx[role_repo_idx[mask]]),
            )

    _push_edges(
        np.take(repo_ids, rule_repo_idx),
        rule_ids,
        "GHHasBranchRule",
        False,
        np.take(org_ids, repo_org_idx[rule_repo_idx]),
    )
    _push_edges(
        np.take(repo_ids, workflow_repo_idx),
        workflow_ids,
        "GHHasWorkflow",
        False,
        np.take(org_ids, repo_org_idx[workflow_repo_idx]),
    )
    _push_edges(
        secrets_owner_id,
        secret_ids,
        "GHHasSecret",
        False,
        secrets_owner_id,
    )

    src = np.concatenate(edge_src)
    dst = np.concatenate(edge_dst)
    kinds = np.concatenate(edge_kind)
    travs = np.concatenate(edge_traversable)
    envs = np.concatenate(edge_env)
    n_edges = src.size
    edges_tbl = pa.table(
        {
            "edge_id": np.array([f"e_{i:09d}" for i in range(n_edges)]),
            "src_id": src,
            "dst_id": dst,
            "edge_kind": kinds,
            "traversable": travs,
            "source_kind": np.full(n_edges, "github"),
            "environment_id": envs,
            "tenant_id": np.full(n_edges, "github"),
            "collected_at": np.full(n_edges, collected_at, dtype=np.int64),
            "weight": np.ones(n_edges, dtype=np.float64),
        }
    )
    pq.write_table(edges_tbl, out_dir / "bh_edges.parquet")

    print(
        f"[gen_github_bh] scale={scale.name} orgs={scale.n_orgs} "
        f"repos={n_repos_total} users={scale.n_users} teams={scale.n_teams} "
        f"rules={scale.branch_rules_total} workflows={scale.workflows_total} "
        f"secrets={scale.secrets_total} roles={n_role} "
        f"bh_nodes={n_nodes} bh_edges={n_edges}",
        file=sys.stderr,
    )
    print(f"[gen_github_bh] wrote {out_dir}", file=sys.stderr)


def main() -> int:
    scale_name = os.environ.get("BH_GITHUB_SCALE", "bh_sf1")
    if scale_name not in SCALE_PRESETS:
        print(
            f"[gen_github_bh] unsupported scale: {scale_name!r}; available: "
            f"{sorted(SCALE_PRESETS)}",
            file=sys.stderr,
        )
        return 2
    scale = SCALE_PRESETS[scale_name]
    seed_override = os.environ.get("BH_GITHUB_SEED")
    if seed_override:
        scale = ScaleParams(**{**scale.__dict__, "seed": int(seed_override)})

    default_out = REPO_ROOT / "fixture" / "data" / "github" / scale.name / "parquet"
    out_dir = Path(os.environ.get("BH_GITHUB_OUT_DIR", str(default_out))).resolve()
    generate(scale, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
