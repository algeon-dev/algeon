//! Per-principal ownership of the ids the Flight SQL server hands to clients.
//!
//! Authentication establishes *who* is calling; this registry establishes
//! *what they may name*. Every id the server returns to a client — a
//! per-execution `query_id` and a prepared-statement handle — is an opaque
//! bearer reference: any authenticated caller that guesses or observes one
//! could otherwise redeem another user's ticket, cancel their query, read
//! their retained SQL and runtime artifact, or close their prepared
//! statement. Recording the issuing principal and checking it on every
//! lookup by id closes that gap.
//!
//! # Anonymous mode
//!
//! Under [`crate::auth::AuthMode::Disabled`] there is no principal, so every
//! caller shares one anonymous owner and all checks pass. That preserves the
//! existing trusted-network behavior exactly: ownership narrows what an
//! authenticated deployment exposes and never adds a new rejection to a
//! deployment that never authenticated in the first place.

use std::sync::Arc;

use dashmap::DashMap;
use tonic::Status;

use crate::auth::Principal;

/// The principal an id belongs to. `Anonymous` is the single shared owner of
/// every id issued while auth is disabled.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum Owner {
    Anonymous,
    User(Arc<str>),
}

impl Owner {
    pub(super) fn from_principal(principal: Option<&Principal>) -> Self {
        principal.map_or(Self::Anonymous, |principal| Self::User(Arc::from(principal.username.as_str())))
    }
}

/// Ownership registry for client-visible ids, keyed by the id itself.
///
/// Query ids and prepared-statement handles are both UUIDs drawn from the
/// same space and are never compared against each other, so one map keyed by
/// id serves both without ambiguity.
#[derive(Debug, Default)]
pub(super) struct QueryOwnership {
    owners: DashMap<String, Owner>,
}

impl QueryOwnership {
    /// Record `id` as belonging to `owner`.
    pub(super) fn insert(&self, id: String, owner: Owner) {
        self.owners.insert(id, owner);
    }

    /// Authorize `caller` to act on `id`.
    ///
    /// An unregistered id is authorized: ids are registered as they are
    /// issued, and the lookup that follows this check is the authority on
    /// whether the id exists at all. Failing closed here would instead turn
    /// every genuinely unknown id into a permission error, which tells a
    /// caller that an id it guessed is unknown rather than merely not theirs.
    pub(super) fn authorize(&self, id: &str, caller: &Owner) -> Result<(), Status> {
        match self.owners.get(id) {
            Some(owner) if owner.value() != caller => {
                // The message deliberately does not distinguish "owned by
                // someone else" from "does not exist", so an id probe learns
                // nothing either way.
                Err(Status::permission_denied("query id or prepared statement handle is not accessible"))
            },
            _ => Ok(()),
        }
    }

    /// Forget `id`. Called when the id can no longer be redeemed, so the
    /// registry does not outlive the resources it guards.
    pub(super) fn remove(&self, id: &str) {
        self.owners.remove(id);
    }

    /// Drop every id for which `is_live` is false.
    ///
    /// Ownership must not become a second lifetime authority: it has no way
    /// to know when a query id stops being resolvable, so it borrows that
    /// answer from whoever owns the id. Sweeping on the existing maintenance
    /// pass keeps the registry bounded by the live id count rather than by
    /// the number of ids ever issued.
    pub(super) fn retain_live(&self, is_live: impl Fn(&str) -> bool) {
        self.owners.retain(|id, _| is_live(id));
    }

    /// Live owner records. Test-only hook for asserting that ids are
    /// released rather than accumulating.
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.owners.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(name: &str) -> Owner {
        Owner::from_principal(Some(&Principal { username: name.to_string() }))
    }

    #[test]
    fn owner_rejects_a_different_principal() {
        let ownership = QueryOwnership::default();
        ownership.insert("q1".to_string(), user("alice"));

        let status = ownership.authorize("q1", &user("mallory")).unwrap_err();
        assert_eq!(status.code(), tonic::Code::PermissionDenied);
        ownership.authorize("q1", &user("alice")).expect("the issuing principal keeps access");
    }

    #[test]
    fn disabled_auth_shares_one_owner() {
        // Without auth there is no principal to distinguish callers, so the
        // registry must not start rejecting requests that previously worked.
        let ownership = QueryOwnership::default();
        ownership.insert("q1".to_string(), Owner::Anonymous);
        ownership.authorize("q1", &Owner::Anonymous).expect("anonymous callers share every id");
    }

    #[test]
    fn anonymous_and_named_principals_are_distinct_owners() {
        let ownership = QueryOwnership::default();
        ownership.insert("q1".to_string(), Owner::Anonymous);
        assert_eq!(ownership.authorize("q1", &user("alice")).unwrap_err().code(), tonic::Code::PermissionDenied);
    }

    #[test]
    fn unregistered_id_defers_to_the_lookup_that_follows() {
        let ownership = QueryOwnership::default();
        ownership.authorize("never-issued", &user("alice")).expect("unknown ids are resolved by their own lookup");
    }

    #[test]
    fn removed_id_stops_being_owned() {
        let ownership = QueryOwnership::default();
        ownership.insert("q1".to_string(), user("alice"));
        ownership.remove("q1");
        assert_eq!(ownership.len(), 0);
        ownership.authorize("q1", &user("mallory")).expect("a released id is no longer owned by anyone");
    }
}
