export function SidebarSkeleton({ count = 6 }: { count?: number }) {
  return (
    <div aria-hidden="true">
      {Array.from({ length: count }).map((_, i) => (
        <span key={i} className="skel skel--sidebar-item" />
      ))}
    </div>
  );
}

export function MessagesSkeleton() {
  return (
    <div aria-hidden="true">
      <span className="skel skel--title" />
      <span className="skel skel--bubble" />
      <span className="skel skel--bubble-short" />
      <span className="skel skel--bubble" />
    </div>
  );
}
