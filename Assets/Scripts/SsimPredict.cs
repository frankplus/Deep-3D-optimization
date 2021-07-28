using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using static SceneController;

public class SsimPredict : MonoBehaviour
{

    public NNModel cnnModelSource;

    public NNModel ffnModelSource;
    
    // Start is called before the first frame update

    Tensor ComputeProjections() 
    {
        var height = Settings.Height;
        var width = Settings.Width;

        SurfaceCountPlaneProjection[] planes = FindObjectsOfType<SurfaceCountPlaneProjection>();
        foreach (var plane in planes)
			plane.ComputeProjections();

        var projections = new Tensor(1, height, width, planes.Length);
        for(int k=0; k<planes.Length; k++)
        {
            var px = planes[k].Pixels;

            // flip png as pixels are ordered left to right, bottom to top
            System.Array.Reverse(px, 0, px.Length);
            for (int i = 0; i < height; i++)
                System.Array.Reverse(px, i * height, height);

            for(int i=0; i<height; i++)
                for(int j=0; j<width; j++)
                    projections[0, i, j, k] = px[i*width + j][0];
        }

        return projections;
    }

    void Start()
    {
        var model = ModelLoader.Load(cnnModelSource);
        var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);

        // var inputTensor = ComputeProjections();
        worker.Execute(new Tensor(4, 64, 64, 3));

        // var inputTensor = new Tensor(1, 2, new float[2] { 0, 0 });
        // worker.Execute(inputTensor);

        // var output = worker.PeekOutput();
        // print("This is the output: " + (output[0] < 0.5? 0 : 1));
        
        // inputTensor.Dispose();
        // output.Dispose();
        worker.Dispose();
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
