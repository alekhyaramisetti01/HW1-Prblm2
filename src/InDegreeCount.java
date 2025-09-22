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

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: InDegreeCount <input> <output>");
            System.exit(1);
        }
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "HW1-P2-InDegree(include zeros)");
        job.setJarByClass(InDegreeCount.class);

        job.setMapperClass(InDegreeMapper.class);
        job.setCombinerClass(InDegreeReducer.class);
        job.setReducerClass(InDegreeReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));    // file or folder
        FileOutputFormat.setOutputPath(job, new Path(args[1]));  // must NOT exist

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
