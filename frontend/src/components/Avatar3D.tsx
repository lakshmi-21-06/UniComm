
import { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, useGLTF, Stage } from "@react-three/drei";

function AvatarModel() {
  const { scene } = useGLTF("/models/avatar.glb");

  return <primitive object={scene} />;
}

export default function Avatar3D() {
  return (
    <div
      style={{
        width: "100%",
        height: "480px",
        overflow: "hidden",
        borderRadius: "16px",
        background: "#f1f0f7",
      }}
    >
      <Canvas camera={{ position: [0, 1.5, 4], fov: 35 }}>
        <ambientLight intensity={0.8} />
        <Suspense fallback={null}>
          <Stage adjustCamera intensity={0.8} environment="city">
            <AvatarModel />
          </Stage>
        </Suspense>
        <OrbitControls makeDefault />
      </Canvas>
    </div>
  );
}

useGLTF.preload("/models/avatar.glb");
