use std::{
    collections::{HashMap, hash_map::Entry},
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use arrow::{
    array::{ArrayRef, BinaryViewArray, DictionaryArray, PrimitiveArray, StringViewArray, UInt64Array, make_array},
    compute::{take, take_record_batch},
    datatypes::{ArrowDictionaryKeyType, ArrowNativeType, Schema},
    error::ArrowError,
    record_batch::{RecordBatch, RecordBatchOptions},
};
use arrow_flight::{FlightData, encode::FlightDataEncoderBuilder, error::FlightError};
use futures::{Stream, task::noop_waker_ref};
use prost::Message;

use super::*;

/// Tonic's default encoded-message limit. Every DoGet path stays below this
/// bound instead of increasing the transport limit.
pub(super) const MAX_FLIGHT_FRAME_BYTES: usize = 4 * 1024 * 1024;

const _: () =
    assert!(MAX_FLIGHT_FRAME_BYTES == cudf::frame::dataframe::HostBatchCursorConfig::DEFAULT_MAX_MATERIALIZATION_BYTES);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CompactProbeObservation {
    rows: usize,
    compact_bytes: usize,
}

struct ActiveBatch {
    batch: RecordBatch,
    offset: usize,
    emitted_empty: bool,
}

/// Row-bounded input adapter for the shared Flight encoder.
///
/// The adapter retains the original batch and constructs at most one compact
/// candidate chunk at a time. Candidate chunks are encoded with the same
/// Flight defaults and checked with `prost::Message::encoded_len`; oversized
/// ranges are reduced with a binary row search.
pub(super) struct FlightFrameLimiter {
    input: Pin<Box<dyn Stream<Item = Result<RecordBatch, FlightError>> + Send>>,
    active: Option<ActiveBatch>,
    max_frame_bytes: usize,
    probe_observer: Option<Arc<dyn Fn(CompactProbeObservation) + Send + Sync>>,
    done: bool,
}

impl FlightFrameLimiter {
    pub(super) fn new<S>(input: S, max_frame_bytes: usize) -> Self
    where S: Stream<Item = Result<RecordBatch, FlightError>> + Send + 'static {
        Self { input: Box::pin(input), active: None, max_frame_bytes, probe_observer: None, done: false }
    }

    #[cfg(test)]
    fn new_observed<S>(
        input: S,
        max_frame_bytes: usize,
        observer: Arc<dyn Fn(CompactProbeObservation) + Send + Sync>,
    ) -> Self
    where
        S: Stream<Item = Result<RecordBatch, FlightError>> + Send + 'static,
    {
        Self { input: Box::pin(input), active: None, max_frame_bytes, probe_observer: Some(observer), done: false }
    }

    fn next_chunk(
        active: &mut ActiveBatch,
        max_frame_bytes: usize,
        probe_observer: Option<&Arc<dyn Fn(CompactProbeObservation) + Send + Sync>>,
    ) -> Result<Option<RecordBatch>, FlightError> {
        if active.batch.num_rows() == 0 {
            if active.emitted_empty {
                return Ok(None);
            }
            active.emitted_empty = true;
            return select_compact_chunk_observed(&active.batch, 0, 0, max_frame_bytes, |observation| {
                if let Some(observer) = probe_observer {
                    observer(observation);
                }
            })
            .map(Some);
        }
        if active.offset == active.batch.num_rows() {
            return Ok(None);
        }

        let remaining = active.batch.num_rows() - active.offset;
        let chunk =
            select_compact_chunk_observed(&active.batch, active.offset, remaining, max_frame_bytes, |observation| {
                if let Some(observer) = probe_observer {
                    observer(observation);
                }
            })?;
        active.offset += chunk.num_rows();
        Ok(Some(chunk))
    }
}

impl Stream for FlightFrameLimiter {
    type Item = Result<RecordBatch, FlightError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if self.done {
                return Poll::Ready(None);
            }
            let max_frame_bytes = self.max_frame_bytes;
            let probe_observer = self.probe_observer.clone();
            if let Some(active) = self.active.as_mut() {
                match Self::next_chunk(active, max_frame_bytes, probe_observer.as_ref()) {
                    Ok(Some(chunk)) => return Poll::Ready(Some(Ok(chunk))),
                    Ok(None) => {
                        self.active = None;
                        continue;
                    },
                    Err(error) => {
                        self.active = None;
                        self.done = true;
                        return Poll::Ready(Some(Err(error)));
                    },
                }
            }

            match self.input.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    self.done = true;
                    return Poll::Ready(None);
                },
                Poll::Ready(Some(Err(error))) => {
                    self.done = true;
                    return Poll::Ready(Some(Err(error)));
                },
                Poll::Ready(Some(Ok(batch))) => {
                    self.active = Some(ActiveBatch { batch, offset: 0, emitted_empty: false });
                },
            }
        }
    }
}

fn select_compact_chunk_observed(
    batch: &RecordBatch,
    offset: usize,
    rows: usize,
    max_frame_bytes: usize,
    mut observe_candidate: impl FnMut(CompactProbeObservation),
) -> Result<RecordBatch, FlightError> {
    if rows == 0 {
        let candidate = compact_batch_range(batch, offset, 0)?;
        observe_candidate(CompactProbeObservation { rows: 0, compact_bytes: compact_batch_memory_size(&candidate) });
        let encoded_bytes = probe_max_encoded_len(&candidate)?;
        return if encoded_bytes <= max_frame_bytes {
            Ok(candidate)
        } else {
            Err(frame_too_large_error(encoded_bytes, max_frame_bytes, 0, batch.num_columns()))
        };
    }

    let mut best_rows = 0usize;
    let mut probe_rows = 1usize;
    let first_oversized_rows = loop {
        let candidate = compact_batch_range(batch, offset, probe_rows)?;
        observe_candidate(CompactProbeObservation {
            rows: probe_rows,
            compact_bytes: compact_batch_memory_size(&candidate),
        });
        let encoded_bytes = probe_max_encoded_len(&candidate)?;
        if encoded_bytes <= max_frame_bytes {
            if probe_rows == rows {
                return Ok(candidate);
            }
            best_rows = probe_rows;
            drop(candidate);
            probe_rows = probe_rows.saturating_mul(2).min(rows);
            continue;
        }
        drop(candidate);
        if probe_rows == 1 {
            return Err(frame_too_large_error(encoded_bytes, max_frame_bytes, 1, batch.num_columns()));
        }
        break probe_rows;
    };

    let mut low = best_rows + 1;
    let mut high = first_oversized_rows - 1;
    while low <= high {
        let middle = low + (high - low) / 2;
        let candidate = compact_batch_range(batch, offset, middle)?;
        observe_candidate(CompactProbeObservation {
            rows: middle,
            compact_bytes: compact_batch_memory_size(&candidate),
        });
        let encoded_bytes = probe_max_encoded_len(&candidate)?;
        drop(candidate);
        if encoded_bytes <= max_frame_bytes {
            best_rows = middle;
            low = middle + 1;
        } else {
            high = middle - 1;
        }
    }

    let candidate = compact_batch_range(batch, offset, best_rows)?;
    observe_candidate(CompactProbeObservation {
        rows: best_rows,
        compact_bytes: compact_batch_memory_size(&candidate),
    });
    Ok(candidate)
}

fn compact_batch_memory_size(batch: &RecordBatch) -> usize {
    batch.columns().iter().map(|column| column.get_array_memory_size()).sum()
}

fn compact_batch_range(batch: &RecordBatch, offset: usize, rows: usize) -> Result<RecordBatch, ArrowError> {
    let end = offset
        .checked_add(rows)
        .ok_or_else(|| ArrowError::InvalidArgumentError("Flight frame chunk row range overflowed usize".to_string()))?;
    if end > batch.num_rows() {
        return Err(ArrowError::InvalidArgumentError(format!(
            "Flight frame chunk range {offset}..{end} exceeds {} rows",
            batch.num_rows()
        )));
    }
    if batch.num_columns() == 0 {
        return RecordBatch::try_new_with_options(
            batch.schema(),
            Vec::new(),
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        );
    }

    let indices = UInt64Array::from_iter_values((offset..end).map(|index| index as u64));
    let taken = take_record_batch(batch, &indices)?;
    let columns = taken.columns().iter().map(compact_view_backing).collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(taken.schema(), columns)
}

/// Recursively garbage-collect variable-width view buffers. `take` already
/// copies ordinary Utf8/Binary backing buffers, but view arrays intentionally
/// preserve all source data buffers and must be compacted explicitly before
/// Arrow IPC serialization.
fn compact_view_backing(array: &ArrayRef) -> Result<ArrayRef, ArrowError> {
    match array.data_type() {
        arrow::datatypes::DataType::Utf8View => {
            let view = array.as_any().downcast_ref::<StringViewArray>().ok_or_else(|| {
                ArrowError::CastError("Utf8View array did not downcast to StringViewArray".to_string())
            })?;
            return Ok(Arc::new(view.gc()));
        },
        arrow::datatypes::DataType::BinaryView => {
            let view = array.as_any().downcast_ref::<BinaryViewArray>().ok_or_else(|| {
                ArrowError::CastError("BinaryView array did not downcast to BinaryViewArray".to_string())
            })?;
            return Ok(Arc::new(view.gc()));
        },
        arrow::datatypes::DataType::Dictionary(key_type, _) => {
            macro_rules! compact_dictionary_key {
                ($key:ty) => {
                    return compact_dictionary::<$key>(array)
                };
            }
            match key_type.as_ref() {
                arrow::datatypes::DataType::Int8 => compact_dictionary_key!(arrow::datatypes::Int8Type),
                arrow::datatypes::DataType::Int16 => compact_dictionary_key!(arrow::datatypes::Int16Type),
                arrow::datatypes::DataType::Int32 => compact_dictionary_key!(arrow::datatypes::Int32Type),
                arrow::datatypes::DataType::Int64 => compact_dictionary_key!(arrow::datatypes::Int64Type),
                arrow::datatypes::DataType::UInt8 => compact_dictionary_key!(arrow::datatypes::UInt8Type),
                arrow::datatypes::DataType::UInt16 => compact_dictionary_key!(arrow::datatypes::UInt16Type),
                arrow::datatypes::DataType::UInt32 => compact_dictionary_key!(arrow::datatypes::UInt32Type),
                arrow::datatypes::DataType::UInt64 => compact_dictionary_key!(arrow::datatypes::UInt64Type),
                other => {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "unsupported Flight dictionary key type {other}"
                    )));
                },
            }
        },
        _ => {},
    }

    let data = array.to_data();
    if data.child_data().is_empty() {
        return Ok(Arc::clone(array));
    }
    let children = data
        .child_data()
        .iter()
        .map(|child| compact_view_backing(&make_array(child.clone())).map(|child| child.to_data()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(make_array(data.into_builder().child_data(children).build()?))
}

fn compact_dictionary<K>(array: &ArrayRef) -> Result<ArrayRef, ArrowError>
where K: ArrowDictionaryKeyType {
    let dictionary = array.as_any().downcast_ref::<DictionaryArray<K>>().ok_or_else(|| {
        ArrowError::CastError(format!("dictionary array did not downcast for key type {}", K::DATA_TYPE))
    })?;
    let mut remapped = HashMap::<usize, usize>::new();
    let mut referenced_values = Vec::<u64>::new();
    let keys = PrimitiveArray::<K>::from_iter(dictionary.keys_iter().map(|key| {
        key.map(|old_key| {
            let next_key = remapped.len();
            let compact_key = match remapped.entry(old_key) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    referenced_values.push(old_key as u64);
                    *entry.insert(next_key)
                },
            };
            K::Native::usize_as(compact_key)
        })
    }));
    let indices = UInt64Array::from_iter_values(referenced_values);
    let values = take(dictionary.values().as_ref(), &indices, None)?;
    let values = compact_view_backing(&values)?;
    Ok(Arc::new(DictionaryArray::<K>::try_new(keys, values)?))
}

fn probe_max_encoded_len(batch: &RecordBatch) -> Result<usize, FlightError> {
    let input = futures::stream::iter([Ok(batch.clone())]);
    let mut encoder = Box::pin(
        FlightDataEncoderBuilder::new().with_schema(batch.schema()).with_max_flight_data_size(usize::MAX).build(input),
    );
    let mut context = Context::from_waker(noop_waker_ref());
    let mut max_encoded_len = 0usize;
    loop {
        match encoder.as_mut().poll_next(&mut context) {
            Poll::Ready(Some(Ok(frame))) => max_encoded_len = max_encoded_len.max(frame.encoded_len()),
            Poll::Ready(Some(Err(error))) => return Err(error),
            Poll::Ready(None) => return Ok(max_encoded_len),
            Poll::Pending => {
                return Err(FlightError::ProtocolError(
                    "in-memory Flight frame size probe unexpectedly became pending".to_string(),
                ));
            },
        }
    }
}

fn frame_too_large_error(
    encoded_frame_bytes: usize,
    max_frame_bytes: usize,
    row_count: usize,
    column_count: usize,
) -> FlightError {
    let status = adapter_error_status(
        Error::new(
            ErrorCode::FlightFrameTooLarge,
            "datafusion_nexus.server.flight.frame_limiter",
            "one Arrow row exceeds the Flight frame size limit and cannot be split safely",
        )
        .with_log_safe_fact("reason", "single_row_exceeds_flight_frame_limit")
        .with_log_safe_fact("encoded_frame_bytes", encoded_frame_bytes)
        .with_log_safe_fact("max_frame_bytes", max_frame_bytes)
        .with_log_safe_fact("row_count", row_count)
        .with_log_safe_fact("column_count", column_count),
    );
    FlightError::from(status)
}

struct ExactFlightFrameGuard {
    input: Pin<Box<dyn Stream<Item = Result<FlightData, FlightError>> + Send>>,
    max_frame_bytes: usize,
    done: bool,
}

impl Stream for ExactFlightFrameGuard {
    type Item = Result<FlightData, FlightError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }
        match self.input.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(frame))) if frame.encoded_len() > self.max_frame_bytes => {
                self.done = true;
                Poll::Ready(Some(Err(frame_too_large_error(frame.encoded_len(), self.max_frame_bytes, 0, 0))))
            },
            Poll::Ready(Some(Err(error))) => {
                self.done = true;
                Poll::Ready(Some(Err(error)))
            },
            Poll::Ready(None) => {
                self.done = true;
                Poll::Ready(None)
            },
            other => other,
        }
    }
}

pub(super) fn encode_flight_data_stream<S>(
    schema: Arc<Schema>,
    input: S,
) -> impl Stream<Item = Result<FlightData, FlightError>> + Send
where
    S: Stream<Item = Result<RecordBatch, FlightError>> + Send + 'static,
{
    let limited = FlightFrameLimiter::new(input, MAX_FLIGHT_FRAME_BYTES);
    let encoded =
        FlightDataEncoderBuilder::new().with_schema(schema).with_max_flight_data_size(usize::MAX).build(limited);
    ExactFlightFrameGuard { input: Box::pin(encoded), max_frame_bytes: MAX_FLIGHT_FRAME_BYTES, done: false }
}

#[cfg(test)]
mod tests {
    use std::{
        ptr::NonNull,
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
        },
    };

    use arrow::{
        array::{Array, BinaryArray, ByteView, Int32Array, StringArray},
        buffer::{Buffer, OffsetBuffer, ScalarBuffer},
        datatypes::{DataType, Field, Int32Type},
    };
    use arrow_flight::{decode::FlightRecordBatchStream, encode::DictionaryHandling};
    use futures::{StreamExt, TryStreamExt};
    use tonic::Code;

    use super::*;

    fn string_batch(values: Vec<String>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Utf8, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(values))]).unwrap()
    }

    #[test]
    fn probing_starts_at_one_row_and_grows_exponentially() {
        let batch = string_batch((0..257).map(|index| index.to_string()).collect());
        let mut observed_rows = Vec::new();
        let chunk = select_compact_chunk_observed(&batch, 0, batch.num_rows(), usize::MAX, |observation| {
            observed_rows.push(observation.rows)
        })
        .unwrap();

        assert_eq!(chunk.num_rows(), batch.num_rows());
        assert_eq!(observed_rows, vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 257]);
    }

    struct TrackedBacking {
        bytes: Vec<u8>,
        dropped: Arc<AtomicBool>,
    }

    impl Drop for TrackedBacking {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn huge_cpu_utf8_batch_probes_from_one_without_full_remaining_copy() {
        const ROWS: usize = 384;
        const ROW_BYTES: usize = 32 * 1024;
        const PROBE_HEADROOM_BYTES: usize = 512 * 1024;

        let dropped = Arc::new(AtomicBool::new(false));
        let mut bytes = Vec::with_capacity(ROWS * ROW_BYTES);
        let mut offsets = Vec::with_capacity(ROWS + 1);
        offsets.push(0_i32);
        for index in 0..ROWS {
            let prefix = format!("{index:08}:");
            bytes.extend_from_slice(prefix.as_bytes());
            bytes.resize(bytes.len() + ROW_BYTES - prefix.len(), b'a' + (index % 26) as u8);
            offsets.push(i32::try_from(bytes.len()).unwrap());
        }
        let backing = Arc::new(TrackedBacking { bytes, dropped: Arc::clone(&dropped) });
        let values_ptr = NonNull::new(backing.bytes.as_ptr().cast_mut()).unwrap();
        let allocation: Arc<dyn arrow::alloc::Allocation> = backing.clone();
        let values = unsafe { Buffer::from_custom_allocation(values_ptr, backing.bytes.len(), allocation) };
        let array = StringArray::new(OffsetBuffer::new(ScalarBuffer::from(offsets)), values, None);
        drop(backing);
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(array)]).unwrap();
        assert!(batch.column(0).get_buffer_memory_size() > 2 * MAX_FLIGHT_FRAME_BYTES);

        let observations = Arc::new(Mutex::new(Vec::<CompactProbeObservation>::new()));
        let observed = Arc::clone(&observations);
        let limiter = FlightFrameLimiter::new_observed(
            futures::stream::iter([Ok(batch)]),
            MAX_FLIGHT_FRAME_BYTES,
            Arc::new(move |observation| observed.lock().unwrap().push(observation)),
        );
        let frames = ExactFlightFrameGuard {
            input: Box::pin(
                FlightDataEncoderBuilder::new()
                    .with_schema(schema)
                    .with_max_flight_data_size(usize::MAX)
                    .build(limiter),
            ),
            max_frame_bytes: MAX_FLIGHT_FRAME_BYTES,
            done: false,
        }
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

        assert!(dropped.load(Ordering::SeqCst), "encoded frames must not retain the original CPU backing");
        assert!(frames.iter().all(|frame| frame.encoded_len() <= MAX_FLIGHT_FRAME_BYTES));
        let observations = observations.lock().unwrap().clone();
        assert_eq!(observations.first().map(|observation| observation.rows), Some(1));
        let max_compact_bytes = observations.iter().map(|observation| observation.compact_bytes).max().unwrap();
        assert!(
            max_compact_bytes <= 2 * MAX_FLIGHT_FRAME_BYTES + ROW_BYTES + PROBE_HEADROOM_BYTES,
            "largest compact candidate {max_compact_bytes} exceeded exponential overshoot bound"
        );
        let decoded = FlightRecordBatchStream::new_from_flight_data(futures::stream::iter(frames.into_iter().map(Ok)))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let mut next_index = 0usize;
        for batch in decoded {
            let values = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
            for value in values.iter().flatten() {
                assert_eq!(value.len(), ROW_BYTES);
                assert!(value.starts_with(&format!("{next_index:08}:")));
                assert_eq!(value.as_bytes()[ROW_BYTES - 1], b'a' + (next_index % 26) as u8);
                next_index += 1;
            }
        }
        assert_eq!(next_index, ROWS);
    }

    #[tokio::test]
    async fn dictionary_keys_and_recursive_view_values_are_compacted_before_transport() {
        let selected = b"selected-dictionary-view".to_vec();
        let dropped = Arc::new(AtomicBool::new(false));
        let unused = Arc::new(TrackedBacking {
            bytes: vec![b'u'; MAX_FLIGHT_FRAME_BYTES + 4096],
            dropped: Arc::clone(&dropped),
        });
        let unused_ptr = NonNull::new(unused.bytes.as_ptr().cast_mut()).expect("non-empty tracked backing");
        let unused_allocation: Arc<dyn arrow::alloc::Allocation> = unused.clone();
        // SAFETY: `unused` owns an immutable allocation at `unused_ptr` for
        // exactly `unused.bytes.len()` bytes and is retained by the Buffer.
        let unused_buffer =
            unsafe { Buffer::from_custom_allocation(unused_ptr, unused.bytes.len(), unused_allocation) };
        let selected_view = ByteView::new(selected.len() as u32, &selected[..4]).with_buffer_index(0).as_u128();
        let unused_view = ByteView::new(unused.bytes.len() as u32, &unused.bytes[..4]).with_buffer_index(1).as_u128();
        let dictionary_values = StringViewArray::new(
            ScalarBuffer::from(vec![selected_view, unused_view]),
            vec![Buffer::from_vec(selected.clone()), unused_buffer],
            None,
        );
        drop(unused);
        let dictionary =
            DictionaryArray::<Int32Type>::try_new(Int32Array::from(vec![Some(0)]), Arc::new(dictionary_values))
                .unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8View)),
            true,
        )]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(dictionary)]).unwrap();
        assert!(batch.column(0).get_buffer_memory_size() > MAX_FLIGHT_FRAME_BYTES);

        let input = futures::stream::iter([Ok(batch)]);
        let frames = ExactFlightFrameGuard {
            input: Box::pin(
                FlightDataEncoderBuilder::new()
                    .with_schema(Arc::new(Schema::new(vec![Field::new(
                        "value",
                        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8View)),
                        true,
                    )])))
                    .with_dictionary_handling(DictionaryHandling::Resend)
                    .with_max_flight_data_size(usize::MAX)
                    .build(FlightFrameLimiter::new(input, MAX_FLIGHT_FRAME_BYTES)),
            ),
            max_frame_bytes: MAX_FLIGHT_FRAME_BYTES,
            done: false,
        }
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

        assert!(dropped.load(Ordering::SeqCst), "transport frames must not retain the unused >4 MiB backing");
        assert!(frames.iter().all(|frame| frame.encoded_len() <= MAX_FLIGHT_FRAME_BYTES));
        let decoded = FlightRecordBatchStream::new_from_flight_data(futures::stream::iter(frames.into_iter().map(Ok)))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(decoded.len(), 1);
        let dictionary = decoded[0]
            .column(0)
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>()
            .unwrap_or_else(|| panic!("decoded dictionary type was {:?}", decoded[0].column(0).data_type()));
        assert_eq!(dictionary.keys().iter().collect::<Vec<_>>(), vec![Some(0)]);
        assert_eq!(dictionary.values().len(), 1);
        let values = dictionary.values().as_any().downcast_ref::<StringViewArray>().unwrap();
        assert_eq!(values.value(0).as_bytes(), selected);
    }

    #[tokio::test]
    async fn exact_limiter_binary_splits_and_round_trips_all_rows() {
        let batch = string_batch((0..128).map(|index| format!("row-{index:04}-{}", "x".repeat(96))).collect());
        let input = futures::stream::iter([Ok(batch.clone())]);
        let frames = ExactFlightFrameGuard {
            input: Box::pin(
                FlightDataEncoderBuilder::new()
                    .with_schema(batch.schema())
                    .with_max_flight_data_size(usize::MAX)
                    .build(FlightFrameLimiter::new(input, 1024)),
            ),
            max_frame_bytes: 1024,
            done: false,
        }
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

        assert!(frames.len() > 2, "schema plus multiple data frames expected");
        assert!(frames.iter().all(|frame| frame.encoded_len() <= 1024));
        let decoded = FlightRecordBatchStream::new_from_flight_data(futures::stream::iter(frames.into_iter().map(Ok)))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(decoded.iter().map(RecordBatch::num_rows).sum::<usize>(), batch.num_rows());
    }

    #[tokio::test]
    async fn sliced_string_view_backing_is_compacted_before_exact_encoding() {
        let values =
            (0..4096).map(|index| format!("long-string-view-{index:04}-{}", "v".repeat(48))).collect::<Vec<_>>();
        let array = StringViewArray::from_iter_values(values.iter().map(String::as_str));
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Utf8View, false)]));
        let full = RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap();
        let sliced = full.slice(2048, 8);
        assert!(sliced.column(0).get_buffer_memory_size() > 100_000);

        let compact = compact_batch_range(&sliced, 0, sliced.num_rows()).unwrap();
        assert!(compact.column(0).get_buffer_memory_size() < 4096);
        assert_eq!(compact.column(0).as_any().downcast_ref::<StringViewArray>().unwrap().value(0), values[2048]);
        let frames = ExactFlightFrameGuard {
            input: Box::pin(
                FlightDataEncoderBuilder::new()
                    .with_schema(compact.schema())
                    .with_max_flight_data_size(usize::MAX)
                    .build(FlightFrameLimiter::new(futures::stream::iter([Ok(sliced)]), 2048)),
            ),
            max_frame_bytes: 2048,
            done: false,
        }
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
        assert!(frames.iter().all(|frame| frame.encoded_len() <= 2048));
    }

    #[tokio::test]
    async fn actual_transport_cap_splits_sliced_string_view_and_round_trips_exactly() {
        let values = (0..160)
            .map(|index| {
                format!("view-{index:04}-{}", char::from(b'a' + (index % 26) as u8).to_string().repeat(65_536))
            })
            .collect::<Vec<_>>();
        let array = StringViewArray::from_iter_values(values.iter().map(String::as_str));
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Utf8View, false)]));
        let full = RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap();
        let sliced = full.slice(32, 96);
        assert!(sliced.column(0).get_buffer_memory_size() > MAX_FLIGHT_FRAME_BYTES);

        let frames =
            encode_flight_data_stream(sliced.schema(), futures::stream::iter([Ok::<_, FlightError>(sliced.clone())]))
                .try_collect::<Vec<_>>()
                .await
                .unwrap();

        let mut schema_frames = 0usize;
        let mut record_batch_frames = 0usize;
        for frame in &frames {
            assert!(
                frame.encoded_len() <= MAX_FLIGHT_FRAME_BYTES,
                "encoded FlightData frame was {} bytes",
                frame.encoded_len()
            );
            let message = arrow::ipc::root_as_message(&frame.data_header).unwrap();
            match message.header_type() {
                arrow::ipc::MessageHeader::Schema => schema_frames += 1,
                arrow::ipc::MessageHeader::RecordBatch => record_batch_frames += 1,
                other => panic!("unexpected FlightData IPC frame {other:?}"),
            }
        }
        assert_eq!(schema_frames, 1);
        assert!(record_batch_frames > 1, "the >4 MiB logical slice must require multiple data frames");

        let decoded = FlightRecordBatchStream::new_from_flight_data(futures::stream::iter(frames.into_iter().map(Ok)))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let actual = decoded
            .iter()
            .flat_map(|batch| {
                let values = batch.column(0).as_any().downcast_ref::<StringViewArray>().unwrap();
                (0..values.len()).map(|row| values.value(row).to_string()).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, values[32..128]);
    }

    #[tokio::test]
    async fn oversized_single_cell_is_structured_resource_exhausted() {
        let batch = string_batch(vec!["x".repeat(8192)]);
        let input = futures::stream::iter([Ok(batch.clone())]);
        let mut stream = FlightFrameLimiter::new(input, 1024);
        let FlightError::Tonic(status) = stream.next().await.unwrap().unwrap_err() else {
            panic!("single-cell overflow must preserve structured tonic status");
        };
        assert_eq!(status.code(), Code::ResourceExhausted);
        assert_eq!(
            status.metadata().get(crate::server::error::ERROR_CODE_METADATA_KEY).and_then(|value| value.to_str().ok()),
            Some("flight_frame_too_large")
        );
    }

    #[tokio::test]
    async fn four_concurrent_oversized_binary_cells_are_rejected_and_limiter_remains_usable() {
        let value = Arc::new(vec![b'x'; 16 * 1024 * 1024]);
        let workers = (0..4)
            .map(|_| {
                let value = Arc::clone(&value);
                tokio::spawn(async move {
                    let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Binary, false)]));
                    let batch = RecordBatch::try_new(schema, vec![Arc::new(BinaryArray::from(vec![value.as_slice()]))])
                        .unwrap();
                    let mut stream =
                        FlightFrameLimiter::new(futures::stream::iter([Ok(batch)]), MAX_FLIGHT_FRAME_BYTES);
                    let FlightError::Tonic(status) = stream.next().await.unwrap().unwrap_err() else {
                        panic!("single-cell binary overflow must preserve structured tonic status");
                    };
                    assert_eq!(status.code(), Code::ResourceExhausted);
                    assert_eq!(
                        status
                            .metadata()
                            .get(crate::server::error::ERROR_CODE_METADATA_KEY)
                            .and_then(|value| value.to_str().ok()),
                        Some("flight_frame_too_large")
                    );
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.await.unwrap();
        }

        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Binary, false)]));
        let small =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(BinaryArray::from(vec![b"ok".as_slice()]))])
                .unwrap();
        let frames = encode_flight_data_stream(schema, futures::stream::iter([Ok::<_, FlightError>(small)]))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert!(frames.iter().all(|frame| frame.encoded_len() <= MAX_FLIGHT_FRAME_BYTES));
    }
}
