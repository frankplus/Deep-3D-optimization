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



public class CameraSnapshotAllLod : MonoBehaviour
{
    public float waitingInterval = 0.001f;
    public string cameraPath;
    public GameObject lodContainer;
    public int resWidth = 1024;
    public int resHeight = 768;
    public string folderScreenshots = "screenshots";
    // optimize for many screenshots will not destroy any objects so future screenshots will be fast
    public bool optimizeForManyScreenshots = true;
    public enum Format { RAW, JPG, PNG, PPM };
    public Format format = Format.PNG;

    // private vars for screenshot
    private Rect rect;
    private RenderTexture renderTexture;
    private Texture2D screenShot;

    private double lastTime;
    private int lastFrameCount;
    private Vector3[] posSequence;
    private Vector3[] orienSequence;
    private int seqNum;
    private StreamWriter logFile;

    private Camera cam;
    private int currentLod = -1;
    private bool running = true;

    void Start()
    {

        Application.runInBackground = true;
        
        cam = this.GetComponent<Camera>();
        
        var lines = File.ReadAllLines(Application.dataPath + "/" + cameraPath);

        posSequence = new Vector3[lines.Length];
        orienSequence = new Vector3[lines.Length];
        for (int i = 0; i < lines.Length; i++)
        {
            string line = lines[i];
            float[] floatData = Array.ConvertAll(line.Split(' '), float.Parse);
            posSequence[i] = new Vector3(floatData[0], floatData[1], floatData[2]);
            orienSequence[i] = new Vector3(floatData[3], floatData[4], floatData[5]);
        }

        NextSequence();

        lastTime = Time.realtimeSinceStartup;
        lastFrameCount = Time.frameCount;
        MoveCam(seqNum);
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

    private void NextSequence()
    {
        if (currentLod == lodContainer.transform.childCount - 1)
        {
            #if UNITY_EDITOR
                UnityEditor.EditorApplication.isPlaying = false;
            #else
                Application.Quit();
            #endif
            running = false;
            return;
        }
        if (currentLod == -1)
            currentLod = 0;
        else
        {
            logFile.Close();
            currentLod++;
        }

        foreach (Transform child in lodContainer.transform)
            child.gameObject.SetActive(false);

        GameObject lodObject = lodContainer.transform.GetChild(currentLod).gameObject;
        string lodName = lodObject.name;
        lodObject.SetActive(true);

        Debug.Log(lodName);

        string folderLog = string.Format("{0}/logData", GetDataPath());
        System.IO.Directory.CreateDirectory(folderLog);
        string fileTmst = string.Format("{0}/{1}.txt", folderLog, lodName);
        logFile = new StreamWriter(fileTmst, false);

        folderScreenshots = string.Format("{0}/screenshots/{1}", GetDataPath(), lodName);
        System.IO.Directory.CreateDirectory(folderScreenshots);

        seqNum = 0;
    }

    void Update()
    {
        double timeInterval = Time.realtimeSinceStartup - lastTime;
        if (timeInterval > waitingInterval)
        {
            // log statistics
            int triCount = UnityEditor.UnityStats.triangles;
            int vertCount = UnityEditor.UnityStats.vertices;
            int textCount = UnityEditor.UnityStats.renderTextureCount;
            double fps_c = (Time.frameCount - lastFrameCount) / timeInterval;

            string line = string.Format("seq_num: {0}; position: {1}; triangle_count: {2}; vertex_count: {3}; textures_count: {4}; fps: {5}",
                            seqNum, cam.transform.position, triCount, vertCount, textCount, fps_c);
            logFile.WriteLine(line);

            TakeScreenshot();

            seqNum++;
            if (seqNum == posSequence.Length)
                NextSequence();

            if (!running)
                return;

            lastTime = Time.realtimeSinceStartup;
            lastFrameCount = Time.frameCount;
            MoveCam(seqNum);
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
        cam.targetTexture = renderTexture;
        cam.Render();

        // read pixels will read from the currently active render texture so make our offscreen 
        // render texture active and then read the pixels
        RenderTexture.active = renderTexture;
        screenShot.ReadPixels(rect, 0, 0);

        // reset active camera texture and render texture
        cam.targetTexture = null;
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

    private string UniqueFilename()
    {
        return string.Format("{0}/{1}.{2}", folderScreenshots, seqNum, format.ToString().ToLower());
    }

    private void MoveCam(int seqNum)
    {
        cam.transform.position = posSequence[seqNum];
        cam.transform.rotation = Quaternion.LookRotation(orienSequence[seqNum] - posSequence[seqNum], Vector3.up);
    }
}

