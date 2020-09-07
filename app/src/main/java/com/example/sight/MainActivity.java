package com.example.sight;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {

    private Bitmap image;
    private Module model;
    private TextView textView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ImageView imageView = findViewById(R.id.image);
        textView = findViewById(R.id.classification);



        try {
            image = BitmapFactory.decodeStream(getAssets().open("image.jpg"));
            String path = assetFilePath(this, "model.pt");
            Log.i("MainActivity", "Until here everything's fine");
            model = Module.load(path);

        }catch (IOException e){
            Log.e("MainActivity", "Error reading assets", e);
        }

        imageView.setImageBitmap(image);

    }

    public static String assetFilePath(Context context, String assetName) throws IOException{
        File file = new File(context.getFilesDir(), assetName);

        if(file.exists() && file.length() > 0)
            return file.getAbsolutePath();

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    public void onClassifyClicked(View view) {

        AsyncTask<Bitmap, Void, String> task = new AsyncTask<Bitmap, Void, String>() {
            @Override
            protected String doInBackground(Bitmap... bitmaps) {

                final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(image,
                        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                        TensorImageUtils.TORCHVISION_NORM_STD_RGB);

                final Tensor outputTensor = model.forward(IValue.from(inputTensor)).toTensor();

                final float[] scores = outputTensor.getDataAsFloatArray();

                float maxScore = -Float.MAX_VALUE;
                int maxScoreIdx = -1;

                for (int i = 0; i < scores.length; i++){
                    if(scores[i] > maxScore){
                        maxScore = scores[i];
                        maxScoreIdx = i;
                    }
                }

                String className = ImageNetClasses.IMAGENET_CLASSES[maxScoreIdx];

                return className;
            }

            @Override
            protected void onPostExecute(String s) {
                textView.setText(s);
            }
        };

        task.execute(image);


    }
}
