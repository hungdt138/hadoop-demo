package com.ptit.dll;

import java.io.IOException;
import java.text.Normalizer;
import java.util.*;
import java.util.regex.Pattern;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

public class InvertedIndex extends Configured implements Tool {

    /** Stopword tiếng Anh + tiếng Việt (theo âm tiết, dạng NFC, chữ thường).
     *  Không đưa vào các âm tiết hay nằm trong từ ghép có nghĩa như "chỉ" (chỉ mục), "từ" (từ khóa). */
    static final Set<String> STOPWORDS = new HashSet<>(Arrays.asList(
            // Tiếng Anh
            "a","an","the","is","are","was","were","be","been","of","to","in","on","at",
            "and","or","but","for","with","as","by","it","this","that","from","i","you",
            "he","she","we","they","his","her","its","not","no","so","if","then","than",
            // Tiếng Việt
            "và","của","là","có","được","các","những","cho","với","trong","này","đó",
            "khi","thì","mà","để","một","đã","đang","sẽ","cũng","như","nên","nhưng",
            "hay","hoặc","rằng","bị","do","vì","nếu","theo","về","rất","nhiều","còn",
            "đến","nào","gì","ở","ấy","nhau","mỗi"));

    /** Mọi ký tự KHÔNG phải chữ cái (L), dấu tổ hợp (M) hay chữ số (N) đều là dấu phân cách. */
    static final Pattern NON_WORD = Pattern.compile("[^\\p{L}\\p{M}\\p{N}]+");

    /** Chuẩn hóa NFC + chữ thường, dùng chung cho văn bản và stopword. */
    static String normalize(String s) {
        return Normalizer.normalize(s, Normalizer.Form.NFC).toLowerCase(Locale.ROOT);
    }

    /** Map: (offset, line) -> (term, "doc:1") */
    public static class IndexMapper extends Mapper<Object, Text, Text, Text> {
        private final Text term = new Text();
        private final Text posting = new Text();
        private String docName;

        @Override
        protected void setup(Context ctx) {
            docName = ((FileSplit) ctx.getInputSplit()).getPath().getName();
        }

        @Override
        public void map(Object key, Text value, Context ctx) throws IOException, InterruptedException {
            String line = NON_WORD.matcher(normalize(value.toString())).replaceAll(" ").trim();
            if (line.isEmpty()) return;
            for (String w : line.split(" ")) {
                // Bỏ 1 ký tự ASCII (a, b, 1...), nhưng giữ âm tiết tiếng Việt 1 ký tự: ý, ổ, ô, ẩ...
                if (w.length() == 1 && w.charAt(0) < 128) continue;
                if (STOPWORDS.contains(w)) continue;
                term.set(w);
                posting.set(docName + ":1");
                ctx.write(term, posting);
            }
        }
    }

    /** Combiner: gộp cục bộ (term, [doc:1, doc:1, ...]) -> (term, doc:n). Cùng định dạng với đầu ra Mapper. */
    public static class IndexCombiner extends Reducer<Text, Text, Text, Text> {
        private final Text out = new Text();
        @Override
        public void reduce(Text key, Iterable<Text> values, Context ctx) throws IOException, InterruptedException {
            Map<String, Integer> counts = sumByDoc(values);
            for (Map.Entry<String, Integer> e : counts.entrySet()) {
                out.set(e.getKey() + ":" + e.getValue());
                ctx.write(key, out);
            }
        }
    }

    /** Reduce: (term, [doc:n, ...]) -> term \t df \t doc1:tf1;doc2:tf2 (sắp theo tf giảm dần) */
    public static class IndexReducer extends Reducer<Text, Text, Text, Text> {
        private final Text out = new Text();
        @Override
        public void reduce(Text key, Iterable<Text> values, Context ctx) throws IOException, InterruptedException {
            Map<String, Integer> counts = sumByDoc(values);
            List<Map.Entry<String, Integer>> list = new ArrayList<>(counts.entrySet());
            list.sort((a, b) -> b.getValue() != a.getValue().intValue()
                    ? b.getValue() - a.getValue() : a.getKey().compareTo(b.getKey()));
            StringBuilder sb = new StringBuilder();
            sb.append(list.size()).append('\t');
            for (int i = 0; i < list.size(); i++) {
                if (i > 0) sb.append(';');
                sb.append(list.get(i).getKey()).append(':').append(list.get(i).getValue());
            }
            out.set(sb.toString());
            ctx.write(key, out);
        }
    }

    static Map<String, Integer> sumByDoc(Iterable<Text> values) {
        Map<String, Integer> counts = new HashMap<>();
        for (Text v : values) {
            String s = v.toString();
            int p = s.lastIndexOf(':');
            counts.merge(s.substring(0, p), Integer.parseInt(s.substring(p + 1)), Integer::sum);
        }
        return counts;
    }

    /** Chạy qua ToolRunner để nhận được tham số -D (vd: -D mapreduce.job.reduces=2). */
    @Override
    public int run(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: com.ptit.dll.InvertedIndex [-D key=value] <input dir> <output dir>");
            return 2;
        }
        Job job = Job.getInstance(getConf(), "Inverted Index");
        job.setJarByClass(InvertedIndex.class);
        job.setMapperClass(IndexMapper.class);
        job.setCombinerClass(IndexCombiner.class);
        job.setReducerClass(IndexReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        return job.waitForCompletion(true) ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        System.exit(ToolRunner.run(new Configuration(), new InvertedIndex(), args));
    }
}