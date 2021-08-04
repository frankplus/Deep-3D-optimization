using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using static SceneController;

public class SsimPredict : MonoBehaviour
{

    public NNModel cnnModelSource;
    public NNModel ffnModelSource;
    public GameObject projectionBox;
    public GameObject lodContainer;
    public float waitingInterval = 0.1f;

    private Camera cam;
    private Tensor cnnFeatures;
    private IWorker ssimPredictorWorker;
    private int counter = 0;
    private double lastTime;
    private int[] allMeshVertexCount;
    private int lodCount;

    private Tensor ComputeProjections() 
    {
        var height = Settings.Height;
        var width = Settings.Width;

        // compute projections
        SurfaceCountPlaneProjection[] planes = FindObjectsOfType<SurfaceCountPlaneProjection>();
        foreach (var plane in planes)
			plane.ComputeProjections();

        // compute order
        var order = new PlaneType[] { PlaneType.HORIZONTAL, PlaneType.VERTICAL, PlaneType.LATERAL };
        var indexOrder = new int[planes.Length];
        for (int i=0; i < planes.Length; i++)
        {
            for (int j = 0; j < planes.Length; j++)
            {
                if (order[i] == planes[j].planeOrientation)
                    indexOrder[i] = j;
            }
        }
        
        // construct tensor
        var projections = new Tensor(1, height, width, planes.Length);
        for(int k=0; k<planes.Length; k++)
        {
            var px = planes[indexOrder[k]].Pixels;

            // flip png as pixels are ordered left to right, bottom to top
            System.Array.Reverse(px, 0, px.Length);
            for (int i = 0; i < height; i++)
                System.Array.Reverse(px, i * height, height);

            for(int i=0; i<height; i++)
                for(int j=0; j<width; j++) {
                    projections[0, i, j, k] = (float) px[i*width + j][0] / 255.0f;
                }
        }

        return projections;
    }

    private Tensor ComputeCnnFeatures()
    {
        var model = ModelLoader.Load(cnnModelSource);
        var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);
        var inputTensor = ComputeProjections();
        worker.Execute(inputTensor);
        var features = worker.PeekOutput().DeepCopy();
        inputTensor.Dispose();
        worker.Dispose();

        return features;
    }

    private (float, float) Predict(Vector3 posRef, Vector3 pos, int lodId)
    {
        float meshVertexCount = Mathf.Log(allMeshVertexCount[lodId], 2) / 30.0f;
        float[] parameters = new float[] {pos.x, pos.y, pos.z, posRef.x, posRef.y, posRef.z, meshVertexCount};
        float[] cnnFeaturesArray = cnnFeatures.ToReadOnlyArray();

        // concatenate
        int batchSize = 1;
        float[] inputArray = new float[batchSize * (parameters.Length + cnnFeaturesArray.Length)];
        parameters.CopyTo(inputArray, 0);
        cnnFeaturesArray.CopyTo(inputArray, parameters.Length);
        Tensor inputTensor = new Tensor(batchSize, 
                                        1, 
                                        1, 
                                        parameters.Length + cnnFeaturesArray.Length, 
                                        inputArray, 
                                        "inputTensor"
                                        );


        this.ssimPredictorWorker.Execute(inputTensor);
        Tensor output = this.ssimPredictorWorker.PeekOutput();
        float ssim = output[0,0,0,0];
        float vertexCount = output[0,0,0,1];
        output.Dispose();
        inputTensor.Dispose();

        return (ssim, vertexCount);
    }

    private float ComputeScore(float ssim, float vertices)
    {
        return 3*ssim + (1-vertices);
    }

    private int PredictBestLod()
    {
        Vector3 posRef = cam.transform.position;
        Vector3 velocity = cam.velocity;
        Vector3 pos = posRef + velocity * 0.2f;

        float maxScore = -1;
        int maxLod = -1;
        for (int i = 0; i < lodCount; i++)
        {
            (float ssim, float vertices) = Predict(posRef, pos, i);
            float score = ComputeScore(ssim, vertices);
            if (score > maxScore)
            {
                maxScore = score;
                maxLod = i;
            }

            Debug.Log(string.Format("lod{0} ssim:{1} vertices:{2} score:{3}", i, ssim, vertices, score));
        }

        return maxLod;
    }

    void Start()
    {
        this.cam = this.GetComponent<Camera>();
        this.cnnFeatures = ComputeCnnFeatures();
        this.projectionBox.SetActive(false);
        this.lodCount = lodContainer.transform.childCount;

        Model model = ModelLoader.Load(ffnModelSource);
        this.ssimPredictorWorker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);

        // init mesh vertex count for each LOD
        this.allMeshVertexCount = new int[lodCount];
        for (int i = 0; i < lodCount; i++)
        {
            GameObject lodObject = lodContainer.transform.GetChild(i).gameObject;
            int vertexCount = lodObject.GetComponentInChildren<MeshFilter>().mesh.vertexCount;
            this.allMeshVertexCount[i] = vertexCount;
            string line = string.Format("lodName: {0}; mesh_vertex_count: {1}", lodObject.name, vertexCount);
            print(line);
        }

        lastTime = Time.realtimeSinceStartup;
    }

    void Update()
    {
        double timeInterval = Time.realtimeSinceStartup - lastTime;
        if (timeInterval > waitingInterval)
        {
            int bestLod = PredictBestLod();
            GameObject bestLodObject = lodContainer.transform.GetChild(bestLod).gameObject;
            foreach (Transform child in lodContainer.transform)
                child.gameObject.SetActive(false);
            bestLodObject.SetActive(true);

            float currentVertices = Mathf.Log(UnityEditor.UnityStats.vertices, 2) / 30f;
            Debug.Log("current number of vertices: " + currentVertices);

            lastTime = Time.realtimeSinceStartup;
        }

        counter++;
    }

    void OnDestroy()
    {
        this.cnnFeatures.Dispose();
        this.ssimPredictorWorker.Dispose();
    }
}
