import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * HW1 – Problem 2: Most Popular Friends (In-Degree)
 *
 * Input line: <userId><tab/space><friend1,friend2,...>
 * Output line: <User>\t<Total_In-Degree>
 *
 * Guarantees:
 *  - Every user that appears on the LEFT side of any line is emitted at least once (with 0),
 *    so users with no incoming edges appear with in-degree 0.
 *
 * Edge cases handled:
 *  - Blank/whitespace-only lines (ignored)
 *  - Lines with just userId (no friends) -> user emitted with 0
 *  - Tab or spaces between user and list
 *  - Extra commas/empty tokens ignored
 *  - Non-numeric tokens ignored
 *  - Negative IDs rejected
 *  - Self-edges (user == friend) ignored
 *  - Duplicate friends on one line counted ONCE (per-record de-dup)
 */
public class InDegreeCount {

    public static enum Counters {
        MALFORMED_LINE,
        MALFORMED_USER_ID,
        EMPTY_FRIEND_LIST,
        NON_NUMERIC_FRIEND_TOKEN,
        SELF_EDGE_SKIPPED,
        DUPLICATE_FRIEND_SKIPPED
    }

    // ----------------------- MAPPER -----------------------
    public static class InDegreeMapper
            extends Mapper<LongWritable, Text, Text, IntWritable> {

        private static final IntWritable ONE = new IntWritable(1);
        private static final IntWritable ZERO = new IntWritable(0);
        private final Text outKey = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context ctx)
                throws IOException, InterruptedException {

            String line = value.toString();
            if (line == null) return;
            line = line.trim();
            if (line.isEmpty()) return;

            // split once on any whitespace into [user, friendsPart]
            String[] parts = line.split("\\s+", 2);
            if (parts.length == 0 || parts[0].isEmpty()) {
                ctx.getCounter(Counters.MALFORMED_LINE).increment(1);
                return;
            }

            int userId;
            try {
                userId = Integer.parseInt(parts[0]);
                if (userId < 0) throw new NumberFormatException();
            } catch (NumberFormatException e) {
                ctx.getCounter(Counters.MALFORMED_USER_ID).increment(1);
                return;
            }

            // Emit ZERO so the reducer will include this user even if nobody lists them
            outKey.set(Integer.toString(userId));
            ctx.write(outKey, ZERO);

            // If no friends part, nothing else to emit
            if (parts.length < 2) {
                ctx.getCounter(Counters.EMPTY_FRIEND_LIST).increment(1);
                return;
            }
            String friendsPart = parts[1].trim();
            if (friendsPart.isEmpty()) {
                ctx.getCounter(Counters.EMPTY_FRIEND_LIST).increment(1);
                return;
            }

            // Deduplicate per record so duplicates on the same line count once
            Set<Integer> uniqueFriends = new HashSet<>();
            String[] toks = friendsPart.split(",");
            for (String tok : toks) {
                if (tok == null) continue;
                tok = tok.trim();
                if (tok.isEmpty()) continue;

                int fid;
                try {
                    fid = Integer.parseInt(tok);
                    if (fid < 0) throw new NumberFormatException();
                } catch (NumberFormatException nfe) {
                    ctx.getCounter(Counters.NON_NUMERIC_FRIEND_TOKEN).increment(1);
                    continue;
                }

                if (fid == userId) {
                    ctx.getCounter(Counters.SELF_EDGE_SKIPPED).increment(1);
                    continue; // ignore self-edge
                }

                // check de-dup
                int before = uniqueFriends.size();
                uniqueFriends.add(fid);
                if (uniqueFriends.size() == before) {
                    ctx.getCounter(Counters.DUPLICATE_FRIEND_SKIPPED).increment(1);
                }
            }

            // Emit +1 for each unique friend -> incoming edge to that friend
            for (Integer fid : uniqueFriends) {
                outKey.set(fid.toString());
                ctx.write(outKey, ONE);
            }
        }
    }

    // ----------------------- REDUCER (also combiner) -----------------------
    public static class InDegreeReducer
            extends Reducer<Text, IntWritable, Text, IntWritable> {

        private final IntWritable outVal = new IntWritable();

        @Override
        protected void reduce(Text key, Iterable<IntWritable> values, Context ctx)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable v : values) sum += v.get();
            outVal.set(sum);
            ctx.write(key, outVal); // <User>\t<Total_In-Degree>
        }
    }

    // ----------------------- DRIVER -----------------------
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: InDegreeCount <input> <output>");
            System.exit(1);
        }

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "HW1-Problem2-InDegree (include zeros)");

        job.setJarByClass(InDegreeCount.class);
        job.setMapperClass(InDegreeMapper.class);
        job.setCombinerClass(InDegreeReducer.class); // safe sum
        job.setReducerClass(InDegreeReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));   // file or folder
        FileOutputFormat.setOutputPath(job, new Path(args[1])); // must NOT already exist

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
