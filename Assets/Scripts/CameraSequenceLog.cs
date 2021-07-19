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
    public float speedCam = 3.0f;

    public string cameraPath;

    public GameObject lodContainer;

    private double lastInterval;
    private Vector3[] pos_sequence;
    private Vector3[] orien_sequence;
    private int seq_num;
    private StreamWriter logFile;

    private Camera cam1;
    private int currentLod = -1;
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

        NextSequence();
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

        seq_num = 0;

        string folderLog = string.Format("{0}/logData/noscreenshot", GetDataPath());
        System.IO.Directory.CreateDirectory(folderLog);

        string fileTmst = string.Format("{0}/{1}.txt", folderLog, lodName);
        logFile = new StreamWriter(fileTmst, false);
    }

    // Update is called once per frame
    void Update()
    {
        double timeNow = Time.realtimeSinceStartup;
        double time_interval = (timeNow - lastInterval);

        if (time_interval > waitingInterval)
        {
            int ipos = (int)(((float)seq_num) / speedCam);
            if (ipos == pos_sequence.Length)
            {
                NextSequence();
                ipos = 0;
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
                string line = string.Format("counter: {0}; seq_num: {1}; position: {2}; triangle_count: {3}; vertex_count: {4}; textures_count: {5}; fps: {6}",
                                ipos, seq_num, cam1.transform.position, triCount, vertCount, textCount, fps_c);
                logFile.WriteLine(line);
            }

            seq_num++;
            lastInterval = timeNow;
        }
    }

    void OnApplicationQuit()
    {
        if (logFile != null)
        {
            logFile.Close();
        }
    }
}

