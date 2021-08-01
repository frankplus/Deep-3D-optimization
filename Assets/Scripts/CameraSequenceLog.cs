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


public class CameraSequenceLog : MonoBehaviour
{
    public float waitingInterval = 0.001f;
    public string cameraPath;
    public GameObject lodContainer;

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

        string folderLog = string.Format("{0}/logData/noscreenshot", GetDataPath());
        System.IO.Directory.CreateDirectory(folderLog);
        string fileTmst = string.Format("{0}/{1}.txt", folderLog, lodName);
        logFile = new StreamWriter(fileTmst, false);

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

    void OnApplicationQuit()
    {
        if (logFile != null)
        {
            logFile.Close();
        }
    }

    private void MoveCam(int seqNum)
    {
        cam.transform.position = posSequence[seqNum];
        cam.transform.rotation = Quaternion.LookRotation(orienSequence[seqNum] - posSequence[seqNum], Vector3.up);
    }
}

