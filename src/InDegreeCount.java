import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Comparator;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.Locale;

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


public class InDegreeCount {

    public static class InDegreeMapper
            extends Mapper<LongWritable, Text, IntWritable, IntWritable> {

        private static final IntWritable ONE = new IntWritable(1);
        private final IntWritable outKey = new IntWritable();

        @Override
        protected void map(LongWritable key, Text value, Context ctx)
                throws IOException, InterruptedException {

            String line = value.toString().trim();
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

           
            if (parts.length < 2 || parts[1].trim().isEmpty()) return;

           
            String[] raw = parts[1].split(",");
            Set<Integer> unique = new HashSet<>(raw.length * 2);
            for (String r : raw) {
                String s = r.trim();
                if (s.isEmpty()) continue;
                try {
                    int fid = Integer.parseInt(s);
                    if (fid < 0 || fid == userId) continue;
                    unique.add(fid);
                } catch (NumberFormatException ignore) {
                }
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
        protected void reduce(IntWritable key, Iterable<IntWritable> vals, Context ctx)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable v : vals) sum += v.get();
            out.set(sum);
            ctx.write(key, out); 
        }
    }

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
        
        final long tAll0 = System.nanoTime();

   
        if (fs.exists(finalDir)) fs.delete(finalDir, true);
        fs.mkdirs(finalDir);

       
        Job job1 = Job.getInstance(conf, "HW1-P2 InDegree (include zeros)");
        job1.setJarByClass(InDegreeCount.class);

     
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

        job1.setNumReduceTasks(1); 
        FileInputFormat.addInputPath(job1, inputPath);
        FileOutputFormat.setOutputPath(job1, tmpOutJob1);

     
        final long tJob10 = System.nanoTime();
        boolean job1Ok = job1.waitForCompletion(true);
        final double job1Sec = (System.nanoTime() - tJob10) / 1e9;
        if (!job1Ok) {
            System.err.println("Job 1 failed.");
            System.exit(1);
        }
        System.out.printf(Locale.US, "[TIME] MR Job1 took %.3f s%n", job1Sec);

     
        Path job1Part = new Path(tmpOutJob1, "part-r-00000");
        Path byUserOut = new Path(finalDir, "by_userid.tsv");
        try (FSDataInputStream in1 = fs.open(job1Part);
             FSDataOutputStream outStream1 = fs.create(byUserOut, true)) {
            IOUtils.copyBytes(in1, outStream1, conf, false);
        }

       
        Path top100Out = new Path(finalDir, "top1000_by_indegree.tsv");
        writeTopK(fs, conf, job1Part, top100Out, 1000);

       
        fs.delete(tmpOutJob1, true);

        System.out.println("  " + byUserOut);
        System.out.println("  " + top100Out);

     
        final double totalSec = (System.nanoTime() - tAll0) / 1e9;
        System.out.printf(Locale.US, "[TIME] MR P2 total took %.3f s%n", totalSec);
    }

   
    private static void writeTopK(FileSystem fs, Configuration conf,
                                  Path srcPart, Path destFile, int k) throws IOException {
        
        PriorityQueue<int[]> heap = new PriorityQueue<>(
            (a, b) -> {
                int c = Integer.compare(a[1], b[1]);   
                if (c != 0) return c;
                return Integer.compare(b[0], a[0]);    
            }
        );

        try (FSDataInputStream in = fs.open(srcPart);
             BufferedReader br = new BufferedReader(new InputStreamReader(in, "UTF-8"))) {
            String line;
            while ((line = br.readLine()) != null) {
                String t = line.trim();
                if (t.isEmpty()) continue;
                String[] parts = t.split("\\s+");
                if (parts.length < 2) continue;

                int u, c;
                try {
                    u = Integer.parseInt(parts[0]);
                    c = Integer.parseInt(parts[1]);
                } catch (NumberFormatException e) {
                    continue;
                }

                heap.offer(new int[]{u, c});
                if (heap.size() > k) heap.poll();
            }
        }

        
        List<int[]> top = new ArrayList<>(heap);
        top.sort((a, b) -> {
            int c = Integer.compare(b[1], a[1]); 
            if (c != 0) return c;
            return Integer.compare(a[0], b[0]);  
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
