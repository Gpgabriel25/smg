//! The build a membership write owes: who pays it and when, what the ask
//! does, and what stays in force at a write's own answer.

use std::{
    collections::HashMap,
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, Instant},
};

use super::{
    ask, block, facts, generation_with, load, member, overloaded, pressure, rank_of, running_of,
    stalled, status, view, Classes, ViewBuffers, ASKER, MODEL, PUBLISH_WINDOW,
};
use crate::{
    store::{
        HealthSink, LoadSink, LoadSource, MemoryStore, WorkerFacts, WorkerStatus, WorkersStore,
    },
    view::{Role, WorkerId},
};

fn worker(index: usize) -> WorkerFacts {
    facts(&format!("http://w{index}:8000"), Role::Single, &[MODEL])
}

/// The workers a decision at `now` may be handed: the pool's routable rows.
fn handed(store: &MemoryStore, now: Instant) -> Vec<WorkerId> {
    let (rows, _, _) = view(store, &ask(now));
    rows.iter()
        .filter(|row| row.health.routable())
        .map(|row| row.id)
        .collect()
}

fn revision(store: &MemoryStore, id: WorkerId) -> u64 {
    WorkersStore::get(store, id).expect("the row").revision
}

#[test]
fn a_registration_owes_its_build_and_the_ask_pays_it_at_once() {
    let store = MemoryStore::new();
    let id = store.upsert(worker(0)).id();
    // The answer is back and the row is on file. No build was paid, so no
    // decision reads the row yet, and the store names when one will.
    assert!(WorkersStore::get(&store, id).is_some());
    let mut out = ViewBuffers::default();
    let now = Instant::now();
    assert!(store
        .build_view(MODEL, Role::Single, &ask(now), &mut out)
        .is_none());
    assert_eq!(store.membership_builds(), 0);
    assert!(store.publish_due().is_some());
    store.publish_now();
    assert_eq!(view(&store, &ask(now)).0.len(), 1);
    assert_eq!((store.membership_builds(), store.publish_due()), (1, None));
    // Nothing is owed: the ask does nothing.
    store.publish_now();
    assert_eq!(store.membership_builds(), 1);
}

#[test]
fn the_writes_of_one_window_are_one_build_at_the_instant_the_store_names() {
    let store = MemoryStore::new();
    store.upsert(worker(0));
    // No window is open: the first build is due at once, and opens one.
    let first = store.publish_due().expect("a build is owed");
    store.publish(first);
    assert_eq!(store.membership_builds(), 1);
    for index in 1..64 {
        store.upsert(worker(index));
    }
    let end = store.publish_due().expect("a build is owed");
    assert!(end >= first + PUBLISH_WINDOW);
    // A call before the instant pays nothing: a storm builds once a window.
    store.publish(end - Duration::from_nanos(1));
    assert_eq!(store.membership_builds(), 1);
    assert_eq!(view(&store, &ask(end)).0.len(), 1);
    store.publish(end);
    assert_eq!(store.membership_builds(), 2);
    assert_eq!(view(&store, &ask(end)).0.len(), 64);
    assert_eq!(store.publish_due(), None);
}

#[test]
fn a_paid_build_opens_the_window_and_an_asked_one_leaves_it_as_it_stands() {
    let store = MemoryStore::new();
    let id = member(&store, "http://a:8000");
    // The ask opened no window: the first record is read at once.
    let t0 = Instant::now();
    store.record(id, 0, load(1, 0), t0, LoadSource::Stream);
    assert_eq!(running_of(&store, id, t0), Some(1));
    // The host pays the next build at the instant the store names. A record
    // inside the window that build opened waits for its end.
    store.upsert(worker(1));
    let due = store.publish_due().expect("a build is owed");
    store.publish(due);
    store.record(id, 0, load(2, 0), due, LoadSource::Stream);
    assert_eq!(running_of(&store, id, due), Some(1));
    store.publish(due + PUBLISH_WINDOW);
    assert_eq!(running_of(&store, id, due), Some(2));
}

#[test]
fn a_record_waits_while_a_build_is_owed_and_is_read_with_it() {
    let store = MemoryStore::new();
    let id = member(&store, "http://a:8000");
    let t0 = Instant::now();
    store.upsert(worker(1));
    // No window is open and the record still waits: the build shows it.
    store.record(id, 0, load(3, 0), t0, LoadSource::Stream);
    assert_eq!(running_of(&store, id, t0), None);
    store.publish_now();
    assert_eq!(running_of(&store, id, t0), Some(3));
}

#[test]
fn a_status_and_a_veto_are_read_at_their_answer_while_a_build_is_owed() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    store.upsert(worker(1));
    status(&store, id, WorkerStatus::Ready);
    assert_eq!(handed(&store, t0), [id]);
    store.gateway_veto(ASKER, id, Some(t0 + Duration::from_secs(1)));
    assert_eq!(stalled(&store, ASKER, t0), [id]);
    // Neither paid the build: the member's own ask made the one.
    assert_eq!(store.membership_builds(), 1);
    assert_eq!(view(&store, &ask(t0)).0.len(), 1);
}

#[test]
fn a_veto_past_its_end_leaves_at_the_build_a_windows_end_pays() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let before = t0 - Duration::from_millis(1);
    let ended = member(&store, "http://a:8000");
    store.gateway_veto(ASKER, ended, Some(t0));
    store.upsert(worker(1));
    assert_eq!(stalled(&store, ASKER, before), [ended]);
    // The host pays the build at or past the veto's end. The build is a
    // window's publication: a decision dated before that end reads it, and
    // it no longer holds the entry.
    let due = store.publish_due().expect("a build is owed");
    store.publish(due);
    assert!(stalled(&store, ASKER, before).is_empty());
    // The sink's own answer keeps its compare.
    assert!(store.vetoed(ASKER, ended, before));
}

#[test]
fn a_leave_is_in_force_at_its_answer_and_its_row_leaves_at_the_build() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let stays = member(&store, "http://a:8000");
    let leaves = member(&store, "http://b:8000");
    for id in [stays, leaves] {
        status(&store, id, WorkerStatus::Ready);
    }
    assert_eq!(handed(&store, t0).len(), 2);
    store
        .remove(leaves, revision(&store, leaves))
        .expect("removed");
    // No build was paid. The pool still lists the row and no decision is
    // handed it: a leave does not wait for the window.
    let (rows, _, _) = view(&store, &ask(t0));
    assert_eq!(rows.len(), 2);
    assert_eq!(handed(&store, t0), [stays]);
    // The row that left reads not ready and draining, both: a pass that
    // reads one of the two bits alone still steps over it.
    let left = rows.iter().find(|row| row.id == leaves).expect("its row");
    assert_eq!((left.health.healthy, left.health.draining), (false, true));
    store.publish_now();
    assert_eq!(view(&store, &ask(t0)).0.len(), 1);
}

#[test]
fn a_replaced_incarnation_leaves_at_the_answer_and_the_new_one_joins_at_the_build() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let old = member(&store, "http://a:8000");
    status(&store, old, WorkerStatus::Ready);
    let new = store
        .upsert(WorkerFacts {
            incarnation: 2,
            ..facts("http://a:8000", Role::Single, &[MODEL])
        })
        .id();
    assert_ne!(new, old);
    // The old life is handed to no decision, and the new one is not built.
    let (rows, _, _) = view(&store, &ask(t0));
    assert_eq!(rows.iter().map(|row| row.id).collect::<Vec<_>>(), [old]);
    assert!(handed(&store, t0).is_empty());
    store.publish_now();
    let (rows, _, _) = view(&store, &ask(t0));
    assert_eq!(rows.iter().map(|row| row.id).collect::<Vec<_>>(), [new]);
}

#[test]
fn a_worker_that_returns_under_its_id_inside_the_window_stays_out_of_the_pools_it_left() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let stays = member(&store, "http://a:8000");
    let leaves = member(&store, "http://b:8000");
    for id in [stays, leaves] {
        status(&store, id, WorkerStatus::Ready);
    }
    store
        .remove(leaves, revision(&store, leaves))
        .expect("removed");
    // The same key and incarnation, registered for another model: the same
    // id, told ready before any build.
    let back = store
        .upsert(facts("http://b:8000", Role::Single, &["other"]))
        .id();
    assert_eq!(back, leaves);
    status(&store, back, WorkerStatus::Ready);
    // The status is the new registration's. The row of the one that left
    // stays out of rotation in the pool it left, and the build reads the
    // status with the registration.
    assert_eq!(handed(&store, t0), [stays]);
    store.publish_now();
    assert_eq!(handed(&store, t0), [stays]);
    let mut out = ViewBuffers::default();
    store
        .build_view("other", Role::Single, &ask(t0), &mut out)
        .expect("the pool it registered for");
    let ready = out.candidates.iter().filter(|row| row.health.routable());
    assert_eq!(ready.map(|row| row.id).collect::<Vec<_>>(), [back]);
}

#[test]
fn a_worker_of_two_pools_that_returns_under_its_id_stays_out_of_both() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let stays = member(&store, "http://a:8000");
    let two = || {
        let models = [MODEL, "other"];
        let facts = facts("http://b:8000", Role::Single, &models);
        store.upsert(facts).id()
    };
    let both = two();
    store.publish_now();
    for id in [stays, both] {
        status(&store, id, WorkerStatus::Ready);
    }
    let other = || {
        let mut out = ViewBuffers::default();
        store
            .build_view("other", Role::Single, &ask(t0), &mut out)
            .expect("the other pool");
        let ready = out.candidates.iter().filter(|row| row.health.routable());
        ready.map(|row| row.id).collect::<Vec<_>>()
    };
    assert_eq!((handed(&store, t0).len(), other()), (2, vec![both]));
    store.remove(both, revision(&store, both)).expect("removed");
    let back = two();
    assert_eq!(back, both);
    status(&store, back, WorkerStatus::Ready);
    // The leave took every place of the worker: the status is the new
    // registration's in each pool, and neither hands it before the build.
    assert_eq!((handed(&store, t0), other()), (vec![stays], vec![]));
    store.publish_now();
    assert_eq!((handed(&store, t0).len(), other()), (2, vec![back]));
}

#[test]
fn no_veto_on_a_worker_that_left_stands_over_the_one_that_returns_under_its_id() {
    let store = MemoryStore::new();
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    status(&store, id, WorkerStatus::Ready);
    store.gateway_veto(ASKER, id, Some(t0 + Duration::from_secs(1)));
    assert_eq!(stalled(&store, ASKER, t0), [id]);
    store.remove(id, revision(&store, id)).expect("removed");
    let back = store
        .upsert(facts("http://a:8000", Role::Single, &[MODEL]))
        .id();
    assert_eq!(back, id);
    status(&store, back, WorkerStatus::Ready);
    store.publish_now();
    // The veto left with the worker it was on, at the leave's answer: the
    // build finds none to put on the one that returned.
    assert_eq!(stalled(&store, ASKER, t0), []);
    assert_eq!(handed(&store, t0), [back]);
}

#[test]
fn the_word_that_no_report_came_leaves_with_the_worker_it_was_said_of() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    store
        .apply(generation_with(1, block(8, 1.0)))
        .expect("applied");
    store.report(&[(id, 0, Some(pressure(9, 0)))], t0, LoadSource::Poll);
    store.report(&[(id, 0, None)], t0, LoadSource::Poll);
    assert_eq!(overloaded(&store, t0), []);
    store.remove(id, revision(&store, id)).expect("removed");
    let back = store
        .upsert(facts("http://a:8000", Role::Single, &[MODEL]))
        .id();
    // The one that returned reports a queue past the threshold before its
    // build. The build judges it from that record: the word said of the one
    // that left is not carried to it.
    store.record(back, 0, pressure(9, 0), t0, LoadSource::Stream);
    store.publish_now();
    assert_eq!(overloaded(&store, t0), [back]);
}

#[test]
fn a_poll_with_no_report_for_a_worker_that_returned_moves_no_verdict_before_its_build() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let id = member(&store, "http://a:8000");
    store
        .apply(generation_with(1, block(8, 1.0)))
        .expect("applied");
    store.report(&[(id, 0, Some(pressure(9, 0)))], t0, LoadSource::Poll);
    assert_eq!(
        (overloaded(&store, t0), store.overload_flips()),
        (vec![id], 1)
    );
    store.remove(id, revision(&store, id)).expect("removed");
    let back = store
        .upsert(facts("http://a:8000", Role::Single, &[MODEL]))
        .id();
    // The poll's word is the new registration's, whose engine is not built
    // yet: it does not reach the engine of the one that left.
    store.report(&[(back, 0, None)], t0, LoadSource::Poll);
    assert_eq!(store.overload_flips(), 1);
}

/// A rank of the two-rank engine, as its registration names it.
fn rank_facts(rank: u16) -> WorkerFacts {
    WorkerFacts {
        engine: "http://e:8000".to_string(),
        dp_rank: rank,
        dp_size: 2,
        ..facts(&format!("http://e:8000#{rank}"), Role::Single, &[MODEL])
    }
}

/// The two ranks of one engine, under a block of eight waiting requests.
fn two_ranks(store: &MemoryStore) -> [WorkerId; 2] {
    let ranks = [0, 1].map(|rank| rank_of(store, "http://e:8000", rank));
    store
        .apply(generation_with(1, block(8, 1.0)))
        .expect("applied");
    ranks
}

/// Rank 0 leaves and registers again under its key, with no build between:
/// the same id.
fn returns(store: &MemoryStore, id: WorkerId) -> WorkerId {
    store.remove(id, revision(store, id)).expect("removed");
    let back = store.upsert(rank_facts(0)).id();
    assert_eq!(back, id);
    back
}

/// Rank 0 registers again as a new life under its key, with no build:
/// another id.
fn restarts(store: &MemoryStore, id: WorkerId) -> WorkerId {
    let life = WorkerFacts {
        incarnation: 2,
        ..rank_facts(0)
    };
    let new = store.upsert(life).id();
    assert_ne!(new, id);
    new
}

/// One rank past the threshold and no row for the other, with both ranks
/// standing and then with rank 0 inside the window `change` opens.
fn partial_poll_after(change: fn(&MemoryStore, WorkerId) -> WorkerId) {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let [a, b] = two_ranks(&store);
    let partial = [(a, 0, Some(pressure(9, 0))), (b, 1, None)];
    store.report(&partial, t0, LoadSource::Poll);
    assert_eq!(overloaded(&store, t0).len(), 2);
    let calm = [(a, 0, Some(pressure(0, 0))), (b, 1, Some(pressure(0, 0)))];
    store.report(&calm, t0, LoadSource::Poll);
    assert_eq!(overloaded(&store, t0), []);
    let again = change(&store, a);
    // The same poll before the build. No row takes the rank's sample, and
    // the word that no report came is said of the rank that stands. The
    // build knows the engine of the rank new to its rows: it carries no
    // word over a row the word's own poll brought, and judges the engine.
    let partial = [(again, 0, Some(pressure(9, 0))), (b, 1, None)];
    store.report(&partial, t0, LoadSource::Poll);
    store.publish_now();
    assert_eq!(overloaded(&store, t0).len(), 2);
}

#[test]
fn the_build_carries_no_word_over_the_row_a_poll_brought_a_returned_rank() {
    partial_poll_after(returns);
}

#[test]
fn the_build_carries_no_word_over_the_row_a_poll_brought_a_restarted_rank() {
    partial_poll_after(restarts);
}

#[test]
fn the_build_carries_no_word_over_the_row_a_poll_brought_a_rank_new_to_its_engine() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let a = rank_of(&store, "http://e:8000", 0);
    store
        .apply(generation_with(1, block(8, 1.0)))
        .expect("applied");
    // The second rank registers after the first was built, and the poll's
    // answer comes inside its window.
    let b = store.upsert(rank_facts(1)).id();
    let partial = [(a, 0, None), (b, 1, Some(pressure(9, 0)))];
    store.report(&partial, t0, LoadSource::Poll);
    store.publish_now();
    assert_eq!(overloaded(&store, t0).len(), 2);
}

/// The engine's word that no report came stands, with a record past the
/// threshold still on file for rank 1.
fn silent_engine(store: &MemoryStore, t0: Instant) -> [WorkerId; 2] {
    let [a, b] = two_ranks(store);
    let loaded = [(a, 0, Some(pressure(0, 0))), (b, 1, Some(pressure(9, 0)))];
    store.report(&loaded, t0, LoadSource::Poll);
    assert_eq!(overloaded(store, t0).len(), 2);
    store.report(&[(a, 0, None), (b, 1, None)], t0, LoadSource::Poll);
    assert_eq!(overloaded(store, t0), []);
    [a, b]
}

/// A record from rank 0 inside the window `change` opens, under its
/// engine's standing word: the build carries no word over it.
fn record_under_a_word_after(change: fn(&MemoryStore, WorkerId) -> WorkerId) {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let [a, _] = silent_engine(&store, t0);
    let again = change(&store, a);
    store.record(again, 0, pressure(0, 0), t0, LoadSource::Stream);
    store.publish_now();
    assert_eq!(overloaded(&store, t0).len(), 2);
}

#[test]
fn the_build_carries_no_word_over_a_returned_ranks_record_since() {
    record_under_a_word_after(returns);
}

#[test]
fn the_build_carries_no_word_over_a_restarted_ranks_record_since() {
    record_under_a_word_after(restarts);
}

#[test]
fn the_build_carries_no_word_over_the_record_of_a_rank_new_to_its_engine() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let a = rank_of(&store, "http://e:8000", 0);
    store
        .apply(generation_with(1, block(8, 1.0)))
        .expect("applied");
    store.report(&[(a, 0, Some(pressure(9, 0)))], t0, LoadSource::Poll);
    assert_eq!(overloaded(&store, t0), [a]);
    store.report(&[(a, 0, None)], t0, LoadSource::Poll);
    assert_eq!(overloaded(&store, t0), []);
    // The second rank registers under the standing word and streams its
    // record before the build.
    let b = store.upsert(rank_facts(1)).id();
    store.record(b, 1, pressure(0, 0), t0, LoadSource::Stream);
    store.publish_now();
    assert_eq!(overloaded(&store, t0).len(), 2);
}

#[test]
fn the_word_that_no_report_came_stays_with_the_rank_whose_sibling_left() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let [a, _] = silent_engine(&store, t0);
    store.remove(a, revision(&store, a)).expect("removed");
    store.publish_now();
    // The engine's last poll brought no report: the build raises no verdict
    // again from the record the rank that stayed has on file.
    assert_eq!(overloaded(&store, t0), []);
}

#[test]
fn a_poll_with_no_report_for_either_rank_says_the_word_after_one_returned() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let [a, b] = two_ranks(&store);
    let loaded = [(a, 0, Some(pressure(0, 0))), (b, 1, Some(pressure(9, 0)))];
    store.report(&loaded, t0, LoadSource::Poll);
    assert_eq!(overloaded(&store, t0).len(), 2);
    let back = returns(&store, a);
    // No row for the rank that returned and none for the one that stayed:
    // the word that no report came is the engine's, and the build carries
    // it over the record still on file.
    store.report(&[(back, 0, None), (b, 1, None)], t0, LoadSource::Poll);
    store.publish_now();
    assert_eq!(overloaded(&store, t0), []);
}

#[test]
fn a_word_keeps_its_polls_instant_across_a_build_and_stands_over_an_older_record() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let [a, b] = silent_engine(&store, t0);
    // A later poll says the word again, and a build of another worker
    // carries it: it keeps the instant that poll began.
    let later = t0 + Duration::from_millis(1);
    store.report(&[(a, 0, None), (b, 1, None)], later, LoadSource::Poll);
    member(&store, "http://other:8000");
    // Rank 0 returns with a record received before that poll began: no
    // news since the word, which the next build carries.
    let back = returns(&store, a);
    store.record(back, 0, pressure(0, 0), t0, LoadSource::Stream);
    store.publish_now();
    assert_eq!(overloaded(&store, t0), []);
}

#[test]
fn the_build_carries_no_word_over_a_restarted_ranks_record_since_the_poll_began() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    let [a, b] = two_ranks(&store);
    let loaded = [(a, 0, Some(pressure(0, 0))), (b, 1, Some(pressure(9, 0)))];
    store.report(&loaded, t0, LoadSource::Poll);
    assert_eq!(overloaded(&store, t0).len(), 2);
    // The rank's new life streams its record while the poll is out, and the
    // poll's answer brings no row before the build. The word is said of
    // the rank that stands; the build finds the record the new id brought
    // since the poll began, and carries none.
    let new = restarts(&store, a);
    store.record(new, 0, pressure(0, 0), t0, LoadSource::Stream);
    store.report(&[(new, 0, None), (b, 1, None)], t0, LoadSource::Poll);
    store.publish_now();
    assert_eq!(overloaded(&store, t0).len(), 2);
}

#[test]
fn a_record_from_another_engines_new_rank_ends_no_word() {
    let store = MemoryStore::with_publish_window(Duration::ZERO);
    let t0 = Instant::now();
    silent_engine(&store, t0);
    // A rank of another engine registers and streams its record before the
    // build: the news is that engine's, and this one's word stays.
    let other = WorkerFacts {
        engine: "http://f:8000".to_string(),
        dp_rank: 0,
        dp_size: 2,
        ..facts("http://f:8000#0", Role::Single, &[MODEL])
    };
    let other = store.upsert(other).id();
    store.record(other, 0, pressure(0, 0), t0, LoadSource::Stream);
    store.publish_now();
    assert_eq!(overloaded(&store, t0), []);
}

#[test]
fn no_decision_is_handed_a_worker_after_its_leaves_answer_under_a_storm() {
    const WORKERS: usize = 256;
    // A window far shorter than the storm: its builds land between leaves.
    let store = MemoryStore::with_publish_window(Duration::from_micros(200));
    let ids: Vec<WorkerId> = (0..WORKERS)
        .map(|index| {
            let id = store.upsert(worker(index)).id();
            store.publish_now();
            status(&store, id, WorkerStatus::Ready);
            id
        })
        .collect();
    let index_of: HashMap<WorkerId, usize> = ids.iter().copied().zip(0..).collect();
    let gone: Vec<AtomicBool> = (0..WORKERS).map(|_| AtomicBool::new(false)).collect();
    let done = AtomicBool::new(false);
    thread::scope(|scope| {
        // The storm: every other worker leaves, by its removal or by a new
        // life under its key, and a new worker registers after each removal.
        scope.spawn(|| {
            for index in (0..WORKERS).step_by(2) {
                if index % 4 == 0 {
                    store
                        .remove(ids[index], revision(&store, ids[index]))
                        .expect("removed");
                    gone[index].store(true, Ordering::SeqCst);
                    store.upsert(worker(WORKERS + index));
                } else {
                    store.upsert(WorkerFacts {
                        incarnation: 2,
                        ..worker(index)
                    });
                    gone[index].store(true, Ordering::SeqCst);
                }
            }
            done.store(true, Ordering::SeqCst);
        });
        // The host, as the driver's loop: it pays what is due.
        scope.spawn(|| {
            while !done.load(Ordering::SeqCst) {
                let now = Instant::now();
                if store.publish_due().is_some_and(|due| now >= due) {
                    store.publish(now);
                }
                thread::yield_now();
            }
        });
        for _ in 0..2 {
            scope.spawn(|| {
                let mut out = ViewBuffers::default();
                let mut left = vec![false; WORKERS];
                while !done.load(Ordering::SeqCst) {
                    // The leaves answered before this decision began.
                    for (left, gone) in left.iter_mut().zip(&gone) {
                        *left = gone.load(Ordering::SeqCst);
                    }
                    store
                        .build_view(MODEL, Role::Single, &ask(Instant::now()), &mut out)
                        .expect("the pool");
                    for row in out.candidates.iter().filter(|row| row.health.routable()) {
                        if let Some(&index) = index_of.get(&row.id) {
                            assert!(!left[index], "worker {index}: handed after its leave");
                        }
                    }
                }
            });
        }
    });
    // Every write is in the last build: the workers that stayed, the new
    // lives, which keep the status of the old, and the new workers, which
    // are pending.
    store.publish_now();
    let now = Instant::now();
    assert_eq!(view(&store, &ask(now)).0.len(), WORKERS);
    assert_eq!(handed(&store, now).len(), WORKERS / 2 + WORKERS / 4);
}

#[test]
fn the_writes_one_build_pays_ask_the_pair_classes_once() {
    let (a, b, c) = (WorkerId(1), WorkerId(2), WorkerId(3));
    let written = |left, joined| Classes::Written { left, joined };
    // A write that moved no offer changes nothing of what another asks.
    let joins = written(None, Some(a));
    assert_eq!(Classes::Carry.then(joins), joins);
    assert_eq!(joins.then(Classes::Carry), joins);
    // One worker's writes stay one worker's: what the last build held of
    // it, and what stands of it now.
    let moved = written(Some(a), Some(a));
    assert_eq!(moved.then(moved), moved);
    assert_eq!(
        written(Some(a), Some(b)).then(written(Some(b), Some(c))),
        written(Some(a), Some(c))
    );
    assert_eq!(
        joins.then(written(Some(a), Some(b))),
        written(None, Some(b))
    );
    assert_eq!(
        written(Some(a), Some(b)).then(written(Some(b), None)),
        written(Some(a), None)
    );
    // A worker that joined and left between two builds was in neither.
    assert_eq!(joins.then(written(Some(a), None)), Classes::Carry);
    // Two workers' writes derive whole, and so does every write after.
    assert_eq!(joins.then(written(None, Some(b))), Classes::Derive);
    assert_eq!(written(Some(a), None).then(joins), Classes::Derive);
    assert_eq!(Classes::Derive.then(Classes::Carry), Classes::Derive);
    assert_eq!(Classes::Derive.then(joins), Classes::Derive);
}

/// The contract names the window by its figure, for a caller that reads no
/// source: the sentence and the store's constant move together.
#[test]
fn the_contract_names_the_window_a_registration_is_in_force_within() {
    const CONTRACT: &str = include_str!("../../../../contracts/policy_controller.proto");
    let figure = format!(
        "publication window ({} ms) of their answer",
        PUBLISH_WINDOW.as_millis()
    );
    assert!(
        CONTRACT.contains(&figure),
        "the contract does not say: {figure}"
    );
}
