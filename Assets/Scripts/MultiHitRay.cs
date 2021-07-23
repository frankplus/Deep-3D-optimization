using UnityEngine;
using System.Collections;
using System.Collections.Generic;

[ExecuteInEditMode]
public class MultiHitRay : MonoBehaviour
{
	[Range(0.1f, 30f)]
	public float Length = 20;
	public LayerMask Mask;
	private List<Vector3> posList = new List<Vector3>();
	int rayCount = 0;

	void Update()
	{
		rayCount = 0;
		posList.Clear();
		RaycastHit hit;

		Vector3 hitPoint = transform.position;

		while (Physics.Raycast(hitPoint, transform.forward, out hit, Length) && rayCount < 100)         // count < 100 Just in case you accidentally enter an infinite loop
		{
			posList.Add(hitPoint);
			rayCount++;
			hitPoint = hit.point + (transform.forward / 100.0f);
		}

		hitPoint = transform.position + transform.forward * Length;

		while (Physics.Raycast(hitPoint, -transform.forward, out hit, Length) && rayCount < 100)
		{
			posList.Add(hitPoint);
			rayCount++;
			hitPoint = hit.point + (-transform.forward / 100.0f);
		}
	}

	void OnGUI()
	{
		GUILayout.Label(string.Format("point count{0} raycount:{1}", posList.Count, rayCount));
	}

	void OnDrawGizmos()
	{
		if (posList == null)
		{
			return;
		}
		Gizmos.color = Color.white;
		Gizmos.DrawSphere(transform.position, 0.1f);
		Gizmos.DrawSphere(transform.position + transform.forward * Length, 0.1f);
		Gizmos.DrawLine(transform.position, transform.position + transform.forward * Length);
		Gizmos.color = Color.red;
		foreach (Vector3 pos in posList)
		{
			Gizmos.DrawSphere(pos, 0.1f);
		}
		Gizmos.color = Color.white;
	}
}