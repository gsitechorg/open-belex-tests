# Should be TEN of these
# APL_OPTIMIZATIONS: Dict[str, Callable] = {
#     "eliminate-read-after-write":       peephole_eliminate_read_after_write,
#     "delete-dead-writes":               delete_dead_writes,
#     "replace-zero-xor" :                peephole_replace_zero_xor,
#     "eliminate-write-read-dependence" : peephole_eliminate_write_read_dependence,
#     "coalesce-sb-from-rl" :             peephole_merge_rl_from_src_and_sb_from_rl,
#     "merge-rl-src-sb" :                 peephole_merge_rl_from_src_and_rl_from_sb,
#     "coalesce-consecutive-and-reads" :  peephole_coalesce_consecutive_and_assignments,
#     "coalesce-sb-from-src" :            peephole_coalesce_two_consecutive_sb_from_src,
#     "coalesce-consecutive-writes" :     peephole_coalesce_two_consecutive_sb_from_rl,
#     "merge-rl-src-sb2" :                peephole_coalesce_shift_before_op,
# }

COMMAND="${PWD}/bin/belex-test"

PLATFORM='-t sim'

OPTIMIZATIONS='-f eliminate-read-after-write -f delete-dead-writes -f coalesce-consecutive-writes -f coalesce-consecutive-and-reads -f coalesce-sb-from-src -f coalesce-sb-from-rl -f merge-rl-src-sb -f merge-rl-src-sb2 -f replace-zero-xor -f eliminate-write-read-dependence'

OUTFILE_PREFIX='develop-branch_OFF'

FILTER="grep 'PASSED\|SUCCESS\|FAILURE\|FATAL\|SYSERR\|hostdrvapi'"

WHOLE="${COMMAND} ${PLATFORM} ${OPTIMIZATIONS} 2>&1 | tee ${OUTFILE_PREFIX}_${i}.txt | ${FILTER}"

SPECIFIC_TEST="tests/test_belex_basic_tests.py::add_u16"

for i in {1..1}
do
    echo "${COMMAND} ${PLATFORM} ${OPTIMIZATIONS} ${SPECIFIC_TEST} 2>&1 | tee ${OUTFILE_PREFIX}_${i}.txt | ${FILTER}"

    ${COMMAND} ${PLATFORM} ${OPTIMIZATIONS} ${SPECIFIC_TEST} 2>&1 | tee ${OUTFILE_PREFIX}_${i}.txt | ${FILTER}

done
