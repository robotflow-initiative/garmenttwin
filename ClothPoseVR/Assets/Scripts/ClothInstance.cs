using Obi;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class ClothInstance : MonoBehaviour
{
    public ObiCloth actor;
    private void Awake()
    {
        actor = GetComponentInChildren<ObiCloth>();
    }

    ObiConstraints<ObiPinConstraintsBatch> pinConstraints = null;
    Dictionary<Transform, ObiPinConstraintsBatch> currentPinConstraints = new Dictionary<Transform, ObiPinConstraintsBatch>();
    public void AddAttach(Transform point)
    {
        if (currentPinConstraints.ContainsKey(point)) return;

        if (pinConstraints == null)
        {
            pinConstraints = actor.GetConstraintsByType(Oni.ConstraintType.Pin) as ObiConstraints<ObiPinConstraintsBatch>;
        }
        ObiCollider collider = point.GetComponent<ObiCollider>() ?? point.gameObject.AddComponent<ObiCollider>();

        Dictionary<int, float> distances = new Dictionary<int, float>();
        for (int i = 0; i < actor.particleCount; i++)
        {
            i = actor.GetParticleRuntimeIndex(i);
            float dis = Vector3.Distance(actor.GetParticlePosition(i), point.position);
            distances.Add(i, dis);
        }

        var order = distances.OrderBy(s => s.Value).Take(5).ToList();
        //if (order[0].Value > 0.2f)
        //    return;
        var batch = new ObiPinConstraintsBatch();
        foreach (var particleIndex in order)
        {
            Vector3 particlePosition = actor.GetParticlePosition(particleIndex.Key);
            Vector3 colliderPosition = collider.transform.InverseTransformPoint(particlePosition);
            Quaternion particleRotation = actor.GetParticleOrientation(particleIndex.Key);
            Quaternion colliderRotation = Quaternion.Inverse(collider.transform.rotation) * particleRotation;
            batch.AddConstraint(particleIndex.Key, collider, colliderPosition, particleRotation, 0, 0, float.PositiveInfinity);
            batch.activeConstraintCount++;
        }
        pinConstraints.AddBatch(batch);
        actor.SetConstraintsDirty(Oni.ConstraintType.Pin);

        currentPinConstraints.Add(point, batch);
    }

    public void RemoveAttach(Transform point)
    {
        if (point == null) return;
        if (currentPinConstraints.TryGetValue(point, out var obiPinConstraintsBatch))
        {
            pinConstraints.RemoveBatch(obiPinConstraintsBatch);
            actor.SetConstraintsDirty(Oni.ConstraintType.Pin);
            currentPinConstraints.Remove(point);
        }
    }

    void GetParticle()
    {
        List<Vector3> particles = new List<Vector3>();
        for (int i = 0; i < actor.particleCount; i++)
        {
            i = actor.GetParticleRuntimeIndex(i);
            particles.Add(actor.GetParticlePosition(i));
        }
    }
}
