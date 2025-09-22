import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


public class InDegreeCount {

   
    public static class InDegreeMapper
            extends Mapper<LongWritable, Text, Text, IntWritable> {

        private static final IntWritable ONE  = new IntWritable(1);
        private static final IntWritable ZERO = new IntWritable(0);
        private final Text outKey = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context ctx)
                throws IOException, InterruptedException {

            String line = value.toString();
            if (line == null) return;
            line = line.trim();
            if (line.isEmpty()) return;

           
            String[] parts = line.split("\\s+", 2);
            if (parts.length == 0 || parts[0].isEmpty()) return;

            int userId;
            try {
                userId = Integer.parseInt(parts[0]);
                if (userId < 0) return;
            } catch (NumberFormatException e) {
                return; 
            }

         
            outKey.set(Integer.toString(userId));
            ctx.write(outKey, ZERO);

            if (parts.length < 2) return;
            String friendsPart = parts[1].trim();
            if (friendsPart.isEmpty()) return;

            
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
                    continue; // ignore non-numeric friend tokens
                }
                if (fid == userId) continue; // ignore self-edge
                unique.add(fid);
            }

            for (Integer fid : unique) {
                outKey.set(fid.toString());
                ctx.write(outKey, ONE); // incoming edge to fid
            }
        }
    }

    
    public static class InDegreeReducer
            extends Reducer<Text, IntWritable, Text, IntWritable> {

        private final IntWritable out = new IntWritable();

        @Override
        protected void reduce(Text key, Iterable<IntWritable> values, Context ctx)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable v : values) sum += v.get();
            out.set(sum);
            ctx.write(key, out);
        }
    }
// ===== SECOND JOB (sort by in-degree, descending) ===========================
class SortByInDegree {

    // Input: lines like "userId<TAB>count" (output of Job 1)
    // Map:   emit (count, userId) so Hadoop sorts by count
    public static class FlipMapper
            extends org.apache.hadoop.mapreduce.Mapper<
                org.apache.hadoop.io.LongWritable,
                org.apache.hadoop.io.Text,
                org.apache.hadoop.io.IntWritable,
                org.apache.hadoop.io.IntWritable> {

        private final org.apache.hadoop.io.IntWritable outKey = new org.apache.hadoop.io.IntWritable();
        private final org.apache.hadoop.io.IntWritable outVal = new org.apache.hadoop.io.IntWritable();

        @Override
        protected void map(org.apache.hadoop.io.LongWritable k,
                           org.apache.hadoop.io.Text v,
                           org.apache.hadoop.mapreduce.Mapper.Context ctx)
                throws java.io.IOException, InterruptedException {

            String line = v.toString().trim();
            if (line.isEmpty()) return;
            String[] parts = line.split("\\s+");
            if (parts.length < 2) return;

            try {
                int user = Integer.parseInt(parts[0]);
                int cnt  = Integer.parseInt(parts[1]);
                outKey.set(cnt);   // sort key = in-degree
                outVal.set(user);  // value   = user id
                ctx.write(outKey, outVal);
            } catch (NumberFormatException ignored) {
                // skip malformed rows quietly
            }
        }
    }

    // Reduce: keys (counts) arrive in DESC order; emit user<TAB>count
    public static class EmitReducer
            extends org.apache.hadoop.mapreduce.Reducer<
                org.apache.hadoop.io.IntWritable,
                org.apache.hadoop.io.IntWritable,
                org.apache.hadoop.io.IntWritable,
                org.apache.hadoop.io.IntWritable> {

        @Override
        protected void reduce(org.apache.hadoop.io.IntWritable count,
                              Iterable<org.apache.hadoop.io.IntWritable> users,
                              org.apache.hadoop.mapreduce.Reducer.Context ctx)
                throws java.io.IOException, InterruptedException {
            for (org.apache.hadoop.io.IntWritable user : users) {
                ctx.write(user, count);
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: SortByInDegree <input_from_job1> <output_sorted>");
            System.exit(1);
        }

        org.apache.hadoop.conf.Configuration conf = new org.apache.hadoop.conf.Configuration();
        org.apache.hadoop.mapreduce.Job job = org.apache.hadoop.mapreduce.Job.getInstance(conf, "HW1-P2-SortByInDegree");

        job.setJarByClass(SortByInDegree.class);
        job.setMapperClass(FlipMapper.class);
        job.setReducerClass(EmitReducer.class);

        job.setMapOutputKeyClass(org.apache.hadoop.io.IntWritable.class);
        job.setMapOutputValueClass(org.apache.hadoop.io.IntWritable.class);
        job.setOutputKeyClass(org.apache.hadoop.io.IntWritable.class);
        job.setOutputValueClass(org.apache.hadoop.io.IntWritable.class);

        // sort by count DESC
        job.setSortComparatorClass(org.apache.hadoop.io.IntWritable.DecreasingComparator.class);

        // one reducer -> one globally sorted file
        job.setNumReduceTasks(1);

        org.apache.hadoop.mapreduce.lib.input.FileInputFormat.addInputPath(job, new org.apache.hadoop.fs.Path(args[0]));
        org.apache.hadoop.mapreduce.lib.output.FileOutputFormat.setOutputPath(job, new org.apache.hadoop.fs.Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
