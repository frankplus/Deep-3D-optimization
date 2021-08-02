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
    public int skipNFrames = 60;

    private Camera cam;
    private Tensor cnnFeatures;
    private IWorker ssimPredictorWorker;
    private int counter = 0;

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

    private (float, float) Predict(Vector3 posRef, Vector3 pos, float lodId)
    {
        float[] parameters = new float[] {pos.x, pos.y, pos.z, posRef.x, posRef.y, posRef.z, lodId};
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

    void Start()
    {
        this.cam = this.GetComponent<Camera>();
        this.cnnFeatures = ComputeCnnFeatures();
        projectionBox.SetActive(false);

        Model model = ModelLoader.Load(ffnModelSource);
        this.ssimPredictorWorker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);
    }

    void Update()
    {
        if (counter % skipNFrames == 0)
        {
            Vector3 posRef = cam.transform.position;
            Vector3 velocity = cam.velocity;
            Vector3 pos = posRef + velocity * 0.5f;
            double fps = 1.0 / Time.deltaTime;

            string[] outputsPerLod = new string[4];
            for (int i=0; i<4; i++)
            {
                (float ssim, float vertexCount) = Predict(posRef, pos, i);
                outputsPerLod[i] = string.Format("{0} {1}", ssim, vertexCount);
            }
            string log = string.Join(" - ", outputsPerLod);
            Debug.Log(log);

            Debug.Log(string.Format("vertex count: {0}", UnityEditor.UnityStats.vertices / 10e7f));
        }

        counter++;
    }

    void OnDestroy()
    {
        this.cnnFeatures.Dispose();
        this.ssimPredictorWorker.Dispose();
    }
}
