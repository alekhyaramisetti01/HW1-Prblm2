import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

/**
 * Run once:
 *   hadoop jar indegree.jar InDegreeCount <inputPath> <OutputFolder>
 *
 * Produces in HDFS:
 *   <OutputFolder>/by_userid.tsv            (Job 1)
 *   <OutputFolder>/top10_by_indegree.tsv    (Job 2, only top 10)
 */
public class InDegreeCount {

    // ===================== Job 1: In-degree =====================
    public static class InDegreeMapper
            extends Mapper<LongWritable, Text, IntWritable, IntWritable> {
        private static final IntWritable ONE  = new IntWritable(1);
        private static final IntWritable ZERO = new IntWritable(0);
        private final IntWritable outKey = new IntWritable();

        @Override
        protected void map(LongWritable key, Text value, Context ctx)
                throws IOException, InterruptedException {
            String line = value.toString();
            if (line == null) return;
            line = line.trim();
            if (line.isEmpty()) return;

            // [userId][whitespace][friend1,friend2,...]  (whitespace may be tab or spaces)
            String[] parts = line.split("\\s+", 2);
            if (parts.length == 0 || parts[0].isEmpty()) return;

            int userId;
            try {
                userId = Integer.parseInt(parts[0]);
                if (userId < 0) return;
            } catch (NumberFormatException e) {
                return;
            }

            // ensure left user appears even with zero in-degree
            outKey.set(userId);
            ctx.write(outKey, ZERO);

            if (parts.length < 2) return;
            String friendsPart = parts[1].trim();
            if (friendsPart.isEmpty()) return;

            // per-record de-dup; ignore self & bad tokens
            Set<Integer> unique = new HashSet<>();
            for (String tok : friendsPart.split(",")) {
                if (tok == null) continue;
                tok = tok.trim();
                if (tok.isEmpty()) continue;
                int fid;
                try {
                    fid = Integer.parseInt(tok);
                    if (fid < 0) continue;
                } catch (NumberFormatException e) {
                    continue;
                }
                if (fid == userId) continue; // ignore self-edge
                unique.add(fid);
            }

            for (Integer fid : unique) {
                outKey.set(fid);
                ctx.write(outKey, ONE); // incoming edge to fid
            }
        }
    }

    public static class InDegreeReducer
            extends Reducer<IntWritable, IntWritable, IntWritable, IntWritable> {
        private final IntWritable out = new IntWritable();
        @Override
        protected void reduce(IntWritable key, Iterable<IntWritable> vals, Context ctx)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable v : vals) sum += v.get();
            out.set(sum);
            ctx.write(key, out); // userId \t inDegree
        }
    }

    // ===================== Job 2: Sort by in-degree (DESC) =====================
    public static class SortByInDegree {

        /** Portable descending comparator for IntWritable keys. */
        public static class DescIntComparator extends WritableComparator {
            protected DescIntComparator() { super(IntWritable.class, true); }
            @Override
            public int compare(WritableComparable a, WritableComparable b) {
                IntWritable x = (IntWritable) a;
                IntWritable y = (IntWritable) b;
                return Integer.compare(y.get(), x.get()); // reverse order
            }
        }

        /** Flip (user,count) -> (count,user) so Hadoop sorts by count. */
        public static class FlipMapper
                extends Mapper<LongWritable, Text, IntWritable, IntWritable> {
            private final IntWritable outKey = new IntWritable();
            private final IntWritable outVal = new IntWritable();
            @Override
            protected void map(LongWritable k, Text v, Context ctx)
                    throws IOException, InterruptedException {
                String line = v.toString().trim();
                if (line.isEmpty()) return;
                String[] parts = line.split("\\s+");
                if (parts.length < 2) return;
                try {
                    int user = Integer.parseInt(parts[0]);
                    int cnt  = Integer.parseInt(parts[1]);
                    outKey.set(cnt);   // key to sort on (DESC via comparator)
                    outVal.set(user);  // value: user id
                    ctx.write(outKey, outVal);
                } catch (NumberFormatException ignored) { }
            }
        }

        /** Emit only the first 10 (globally) because we use 1 reducer. */
        public static class EmitReducer
                extends Reducer<IntWritable, IntWritable, IntWritable, IntWritable> {

            private final IntWritable outUser  = new IntWritable();
            private final IntWritable outCount = new IntWritable();
            private int emitted = 0; // total rows written

            @Override
            protected void reduce(IntWritable count, Iterable<IntWritable> users, Context ctx)
                    throws IOException, InterruptedException {
                if (emitted >= 10) return; // already got top 10

                for (IntWritable user : users) {
                    if (emitted >= 10) break;
                    outUser.set(user.get());      // userId
                    outCount.set(count.get());    // in-degree
                    ctx.write(outUser, outCount); // userId \t inDegree
                    emitted++;
                }
            }
        }
    }

    // ===================== One-run driver =====================
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: InDegreeCount <inputPath> <OutputFolder>");
            System.exit(1);
        }

        Configuration conf = new Configuration();
        Path inputPath = new Path(args[0]);
        Path finalDir  = new Path(args[1]);
        Path tmpOutJob1 = new Path(finalDir, "_tmp_job1");
        Path tmpOutJob2 = new Path(finalDir, "_tmp_job2");

        FileSystem fs = FileSystem.get(conf);

        // Clean & create parent OutputFolder so tmp dirs can live under it
        if (fs.exists(finalDir)) fs.delete(finalDir, true);
        fs.mkdirs(finalDir);

        // -------- Job 1: In-degree (include zeros) --------
        Job job1 = Job.getInstance(conf, "HW1-P2 InDegree (include zeros)");
        job1.setJarByClass(InDegreeCount.class);

        // force tab separator to avoid newline-between-key/value
        job1.setOutputFormatClass(TextOutputFormat.class);
        job1.getConfiguration().set("mapreduce.output.textoutputformat.separator", "\t");
        job1.getConfiguration().set("mapred.textoutputformat.separator", "\t");
        job1.getConfiguration().set("mapreduce.output.key.field.separator", "\t");

        job1.setMapperClass(InDegreeMapper.class);
        job1.setCombinerClass(InDegreeReducer.class);
        job1.setReducerClass(InDegreeReducer.class);

        job1.setMapOutputKeyClass(IntWritable.class);
        job1.setMapOutputValueClass(IntWritable.class);
        job1.setOutputKeyClass(IntWritable.class);
        job1.setOutputValueClass(IntWritable.class);

        job1.setNumReduceTasks(1); // single globally userId-sorted file
        FileInputFormat.addInputPath(job1, inputPath);
        FileOutputFormat.setOutputPath(job1, tmpOutJob1);

        if (!job1.waitForCompletion(true)) {
            System.err.println("Job 1 failed.");
            System.exit(1);
        }

        // -------- Job 2: Sort by in-degree (desc) and keep TOP 10 --------
        Job job2 = Job.getInstance(conf, "HW1-P2 SortByInDegree (TOP 10)");
        job2.setJarByClass(SortByInDegree.class);

        job2.setOutputFormatClass(TextOutputFormat.class);
        job2.getConfiguration().set("mapreduce.output.textoutputformat.separator", "\t");
        job2.getConfiguration().set("mapred.textoutputformat.separator", "\t");
        job2.getConfiguration().set("mapreduce.output.key.field.separator", "\t");

        job2.setMapperClass(SortByInDegree.FlipMapper.class);
        job2.setReducerClass(SortByInDegree.EmitReducer.class);

        job2.setMapOutputKeyClass(IntWritable.class);
        job2.setMapOutputValueClass(IntWritable.class);
        job2.setOutputKeyClass(IntWritable.class);
        job2.setOutputValueClass(IntWritable.class);

        job2.setSortComparatorClass(SortByInDegree.DescIntComparator.class);
        job2.setNumReduceTasks(1); // must be 1 so "first 10" is global

        FileInputFormat.addInputPath(job2, tmpOutJob1);
        FileOutputFormat.setOutputPath(job2, tmpOutJob2);

        if (!job2.waitForCompletion(true)) {
            System.err.println("Job 2 failed.");
            System.exit(1);
        }

        // -------- Copy outputs to nice filenames --------
        Path job1Part = new Path(tmpOutJob1, "part-r-00000");
        Path job2Part = new Path(tmpOutJob2, "part-r-00000");
        Path out1 = new Path(finalDir, "by_userid.tsv");
        Path out2 = new Path(finalDir, "top10_by_indegree.tsv");

        try (FSDataInputStream in1 = fs.open(job1Part);
             FSDataOutputStream outStream1 = fs.create(out1, true)) {
            IOUtils.copyBytes(in1, outStream1, conf, false);
        }
        try (FSDataInputStream in2 = fs.open(job2Part);
             FSDataOutputStream outStream2 = fs.create(out2, true)) {
            IOUtils.copyBytes(in2, outStream2, conf, false);
        }

        // Cleanup temp dirs
        fs.delete(tmpOutJob1, true);
        fs.delete(tmpOutJob2, true);

        System.out.println("Done. Final files:");
        System.out.println("  " + out1);
        System.out.println("  " + out2);
    }
}
