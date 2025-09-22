import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


public class InDegreeCount {

   
    public static class InDegreeMapper
            extends Mapper<LongWritable, org.apache.hadoop.io.Text, IntWritable, IntWritable> {

        private static final IntWritable ONE  = new IntWritable(1);
        private static final IntWritable ZERO = new IntWritable(0);
        private final IntWritable outKey = new IntWritable();

        @Override
        protected void map(LongWritable key, org.apache.hadoop.io.Text value, Context ctx)
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
                return; // malformed left id
            }

         
            outKey.set(userId);
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
                    continue;
                }
                if (fid == userId) continue; 
                unique.add(fid);
            }

            for (Integer fid : unique) {
                outKey.set(fid);
                ctx.write(outKey, ONE); 
            }
        }
    }

 
    public static class InDegreeReducer
            extends Reducer<IntWritable, IntWritable, IntWritable, IntWritable> {

        private final IntWritable out = new IntWritable();

        @Override
        protected void reduce(IntWritable key, Iterable<IntWritable> values, Context ctx)
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
        Job job = Job.getInstance(conf, "HW1-P2-InDegree (include zeros)");
        job.setJarByClass(InDegreeCount.class);

        job.setMapperClass(InDegreeMapper.class);
        job.setCombinerClass(InDegreeReducer.class);
        job.setReducerClass(InDegreeReducer.class);

      
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(IntWritable.class);
       
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(IntWritable.class);

       
        job.setNumReduceTasks(1);

        FileInputFormat.addInputPath(job, new Path(args[0]));      
        FileOutputFormat.setOutputPath(job, new Path(args[1]));    

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}


class SortByInDegree {

  
    public static class FlipMapper
            extends Mapper<LongWritable, org.apache.hadoop.io.Text, IntWritable, IntWritable> {

        private final IntWritable outKey = new IntWritable();
        private final IntWritable outVal = new IntWritable();

        @Override
        protected void map(LongWritable k, org.apache.hadoop.io.Text v, Context ctx)
                throws IOException, InterruptedException {
            String line = v.toString().trim();
            if (line.isEmpty()) return;

            String[] parts = line.split("\\s+");
            if (parts.length < 2) return;

            try {
                int user = Integer.parseInt(parts[0]);
                int cnt  = Integer.parseInt(parts[1]);
                outKey.set(cnt);    
                outVal.set(user);   
                ctx.write(outKey, outVal);
            } catch (NumberFormatException ignored) {
                
            }
        }
    }


    public static class EmitReducer
            extends Reducer<IntWritable, IntWritable, IntWritable, IntWritable> {

        @Override
        protected void reduce(IntWritable count, Iterable<IntWritable> users, Context ctx)
                throws IOException, InterruptedException {
            for (IntWritable user : users) {
                ctx.write(user, count);
            }
        }
    }

   
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: SortByInDegree <input_from_job1> <output_sorted>");
            System.exit(1);
        }
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "HW1-P2-SortByInDegree (desc)");

        job.setJarByClass(SortByInDegree.class);
        job.setMapperClass(FlipMapper.class);
        job.setReducerClass(EmitReducer.class);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(IntWritable.class);

       
        job.setSortComparatorClass(IntWritable.DecreasingComparator.class);

        
        job.setNumReduceTasks(1);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
