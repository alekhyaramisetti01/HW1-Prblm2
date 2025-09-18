import java.io.IOException;
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

/**
 * HW1 – Problem 2: Most Popular Friends (In-Degree)
 * Input line: <userId><tab/space><friend1,friend2,...>
 * Output: <friendId>\t<total_in_degree>
 */
public class InDegreeCount {

    // Mapper: for each outgoing friend list, emit <friendId, 1>
    public static class InDegreeMapper
            extends Mapper<LongWritable, Text, Text, IntWritable> {

        private static final IntWritable ONE = new IntWritable(1);
        private final Text friendId = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {

            String line = value.toString().trim();
            if (line.isEmpty()) return;

            // split once on whitespace (tab or spaces) into [user, friends...]
            String[] parts = line.split("\\s+", 2);
            if (parts.length < 2) return; // user with no listed friends

            String friendsPart = parts[1].trim();
            if (friendsPart.isEmpty()) return;

            String[] friends = friendsPart.split(",");
            for (String f : friends) {
                f = f.trim();
                if (!f.isEmpty()) {
                    friendId.set(f);
                    context.write(friendId, ONE); // incoming edge to f
                }
            }
        }
    }

    // Reducer (also used as Combiner): sum counts per friendId
    public static class InDegreeReducer
            extends Reducer<Text, IntWritable, Text, IntWritable> {

        private final IntWritable out = new IntWritable();

        @Override
        protected void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable v : values) sum += v.get();
            out.set(sum);
            context.write(key, out); // <User>\t<Total_In-Degree>
        }
    }

    // Driver
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: InDegreeCount <input> <output>");
            System.exit(1);
        }

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "HW1-Problem2-InDegree");

        job.setJarByClass(InDegreeCount.class);
        job.setMapperClass(InDegreeMapper.class);
        job.setCombinerClass(InDegreeReducer.class); // optimization
        job.setReducerClass(InDegreeReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
