import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Comparator;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.ArrayList;
import java.util.List;
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
 *   <OutputFolder>/by_userid.tsv          (all users with in-degree)
 *   <OutputFolder>/top10_by_indegree.tsv  (top 10 by in-degree, DESC)
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
                return; // ignore malformed left id
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

    // ===================== One-run driver (Job 1 + normal sort) =====================
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: InDegreeCount <inputPath> <OutputFolder>");
            System.exit(1);
        }

        Configuration conf = new Configuration();
        Path inputPath = new Path(args[0]);
        Path finalDir  = new Path(args[1]);
        Path tmpOutJob1 = new Path(finalDir, "_tmp_job1");

        FileSystem fs = FileSystem.get(conf);

        // Clean & create parent OutputFolder so tmp dir can live under it
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

        // -------- Make a nice copy of Job 1 output --------
        Path job1Part = new Path(tmpOutJob1, "part-r-00000");
        Path byUserOut = new Path(finalDir, "by_userid.tsv");
        try (FSDataInputStream in1 = fs.open(job1Part);
             FSDataOutputStream outStream1 = fs.create(byUserOut, true)) {
            IOUtils.copyBytes(in1, outStream1, conf, false);
        }

        // -------- Normal (non-MR) TOP-10 sort over Job 1 output --------
        Path top10Out = new Path(finalDir, "top10_by_indegree.tsv");
        writeTopK(fs, conf, job1Part, top10Out, 10);

        // Cleanup temp dir
        fs.delete(tmpOutJob1, true);

        System.out.println("Done. Final files:");
        System.out.println("  " + byUserOut);
        System.out.println("  " + top10Out);
    }

    /** Read lines "userId<TAB>count" and write top-K by count to dest, sorted DESC. */
    private static void writeTopK(FileSystem fs, Configuration conf,
                                  Path srcPart, Path destFile, int k) throws IOException {
        // Min-heap: smallest count at head; keep at most k elements.
        PriorityQueue<int[]> heap = new PriorityQueue<>(
            Comparator.<int[]>comparingInt(a -> a[1])   // by count ASC
                      .thenComparingInt(a -> a[0])       // then userId ASC (stable tie-break)
        );

        try (FSDataInputStream in = fs.open(srcPart);
             BufferedReader br = new BufferedReader(new InputStreamReader(in, "UTF-8"))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) continue;
                String[] parts = line.split("\\s+");
                if (parts.length < 2) continue;
                try {
                    int user = Integer.parseInt(parts[0]);
                    int cnt  = Integer.parseInt(parts[1]);
                    heap.offer(new int[]{user, cnt});
                    if (heap.size() > k) heap.poll(); // pop smallest
                } catch (NumberFormatException ignored) { }
            }
        }

        // Move heap to list and sort DESC by count, then userId ASC for stable order
        List<int[]> top = new ArrayList<>(heap);
        top.sort((a, b) -> {
            int c = Integer.compare(b[1], a[1]); // count DESC
            if (c != 0) return c;
            return Integer.compare(a[0], b[0]);  // userId ASC
        });

        try (FSDataOutputStream out = fs.create(destFile, true)) {
            StringBuilder sb = new StringBuilder();
            for (int[] row : top) {
                sb.setLength(0);
                sb.append(row[0]).append('\t').append(row[1]).append('\n');
                out.write(sb.toString().getBytes("UTF-8"));
            }
        }
    }
}
