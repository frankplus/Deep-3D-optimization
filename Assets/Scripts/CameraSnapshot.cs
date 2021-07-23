using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.IO;



public class CameraSnapshot : MonoBehaviour
{

    public int resWidth = 3300;
    public int resHeight = 2550;
    public float waitingInterval = 0.001f;
    public float speedCam = 3.0f;

    // folder to write output (defaults to data path)
    public string folderScreenshots;
    public string cameraPath;

    public string sequenceName;

    // private vars for screenshot
    private Rect rect;
    private RenderTexture renderTexture;
    private Texture2D screenShot;
    private double lastInterval;
    private Vector3[] pos_sequence;
    private Vector3[] orien_sequence;
    private int seq_num;
    private string folderLog;
    private StreamWriter logFile;

    // optimize for many screenshots will not destroy any objects so future screenshots will be fast
    public bool optimizeForManyScreenshots = true;

    // configure with raw, jpg, png, or ppm (simple raw format)
    public enum Format { RAW, JPG, PNG, PPM };
    public Format format = Format.RAW;

    private bool takeHiResShot = false;
    private Camera cam1;
    private bool running = true;

    void Start()
    {

        Application.runInBackground = true;
        
        cam1 = this.GetComponent<Camera>();
        
        var lines = File.ReadAllLines(Application.dataPath + "/" + cameraPath);

        pos_sequence = new Vector3[lines.Length];
        orien_sequence = new Vector3[lines.Length];
        for (int i = 0; i < lines.Length; i++)
        {
            string line = lines[i];
            float[] floatData = Array.ConvertAll(line.Split(' '), float.Parse);
            pos_sequence[i] = new Vector3(floatData[0], floatData[1], floatData[2]);
            orien_sequence[i] = new Vector3(floatData[3], floatData[4], floatData[5]);
        }

        InitSequence(sequenceName);
    }

    private string GetDataPath()
    {
        // Init screenshot folder
        string path = Application.dataPath;
        if (Application.isEditor)
        {
            // put screenshots in folder above asset path so unity doesn't index the files
            var stringPath = path + "/..";
            path = Path.GetFullPath(stringPath);
        }
        return path;
    }

    private void InitSequence(string name)
    {
        takeHiResShot = false;
        seq_num = 0;

        folderScreenshots = string.Format("{0}/screenshots/{1}", GetDataPath(), name);
        System.IO.Directory.CreateDirectory(folderScreenshots);

        folderLog = string.Format("{0}/logData", GetDataPath());
        System.IO.Directory.CreateDirectory(folderLog);

        string fileTmst = string.Format("{0}/{1}.txt", folderLog, name);
        logFile = new StreamWriter(fileTmst, false);
    }

    // Update is called once per frame
    void Update()
    {
        double timeNow = Time.realtimeSinceStartup;
        double time_interval = (timeNow - lastInterval);

        takeHiResShot |= (time_interval > waitingInterval);
        if (takeHiResShot)
        {
            int ipos = (int)(((float)seq_num) / speedCam);
            if (ipos == pos_sequence.Length)
            {
                #if UNITY_EDITOR
                    UnityEditor.EditorApplication.isPlaying = false;
                #else
                    Application.Quit();
                #endif
                running = false;
            }

            if (!running)
                return;

            int triCount = UnityEditor.UnityStats.triangles;
            int vertCount = UnityEditor.UnityStats.vertices;
            int textCount = UnityEditor.UnityStats.renderTextureCount;
            double fps_c = 1.0 / Time.deltaTime;

            cam1.transform.position = pos_sequence[ipos];
            cam1.transform.rotation = Quaternion.LookRotation(orien_sequence[ipos] - pos_sequence[ipos], Vector3.up);

            if (logFile != null)
            {
                string timestamp = System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss.fff");
                string line = string.Format("counter: {0}; seq_num: {1}; position: {2}; triangle_count: {3}; vertex_count: {4}; textures_count: {5}; fps: {6}; timestamp: {7}",
                                ipos, seq_num, cam1.transform.position, triCount, vertCount, textCount, fps_c, timestamp);
                logFile.WriteLine(line);
            }

            TakeScreenshot();

            seq_num++;
            takeHiResShot = false;
            lastInterval = timeNow;
        }
    }

    private void TakeScreenshot()
    {
        // create screenshot objects if needed
        if (renderTexture == null)
        {
            // creates off-screen render texture that can rendered into
            rect = new Rect(0, 0, resWidth, resHeight);
            renderTexture = new RenderTexture(resWidth, resHeight, 24);
            screenShot = new Texture2D(resWidth, resHeight, TextureFormat.RGB24, false);
        }

        // get main camera and manually render scene into rt
        cam1.targetTexture = renderTexture;
        cam1.Render();

        // read pixels will read from the currently active render texture so make our offscreen 
        // render texture active and then read the pixels
        RenderTexture.active = renderTexture;
        screenShot.ReadPixels(rect, 0, 0);

        // reset active camera texture and render texture
        cam1.targetTexture = null;
        RenderTexture.active = null;

        // get our unique filename
        string filename = UniqueFilename();

        // pull in our file header/data bytes for the specified image format (has to be done from main thread)
        byte[] fileHeader = null;
        byte[] fileData = null;

        if (format == Format.RAW)
        {
            fileData = screenShot.GetRawTextureData();
        }
        else if (format == Format.PNG)
        {
            fileData = screenShot.EncodeToPNG();
        }
        else if (format == Format.JPG)
        {
            fileData = screenShot.EncodeToJPG();
        }
        else // ppm
        {
            // create a file header for ppm formatted file
            string headerStr = string.Format("P6\n{0} {1}\n255\n", rect.width, rect.height);
            fileHeader = System.Text.Encoding.ASCII.GetBytes(headerStr);
            fileData = screenShot.GetRawTextureData();
        }

        // create new thread to save the image to file (only operation that can be done in background)
        new System.Threading.Thread(() =>
        {
            // create file and write optional header with image bytes
            var f = System.IO.File.Create(filename);
            if (fileHeader != null) f.Write(fileHeader, 0, fileHeader.Length);
            f.Write(fileData, 0, fileData.Length);
            f.Close();
        }).Start();

        // cleanup if needed
        if (optimizeForManyScreenshots == false)
        {
            Destroy(renderTexture);
            renderTexture = null;
            screenShot = null;
        }
    }

    void OnApplicationQuit()
    {
        if (logFile != null)
        {
            logFile.Close();
        }
    }

    // create a unique filename
    private string UniqueFilename()
    {
        return string.Format("{0}/{1}.{2}", folderScreenshots, seq_num, format.ToString().ToLower());
    }

}

