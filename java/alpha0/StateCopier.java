package alpha0;

import java.io.*;
import java.lang.reflect.*;
import java.util.*;
import sun.misc.Unsafe;

/** Exact mutable object-graph copy for the pinned engine, with cyclic references preserved.
 * The engine's precomputed BFS tables and recording-free sink are immutable and shared.
 * Never use this on instrumented benchmark sandboxes (which contain live threads).
 */
public final class StateCopier {
    private static final Unsafe U;
    static {
        try { Field f=Unsafe.class.getDeclaredField("theUnsafe"); f.setAccessible(true); U=(Unsafe)f.get(null); }
        catch(Exception e) { throw new ExceptionInInitializerError(e); }
    }
    private final IdentityHashMap<Object,Object> seen=new IdentityHashMap<>();
    @SuppressWarnings({"unchecked","rawtypes"})
    public <T> T copy(T source) {
        if(source==null) return null;
        Class<?> c=source.getClass();
        if(c.isEnum() || source instanceof String || source instanceof Number || source instanceof Boolean || source instanceof Character || source instanceof Class || source instanceof NoopGameMaker || source instanceof NoopGameMaker.Sink) return source;
        if(seen.containsKey(source)) return (T)seen.get(source);
        try {
            if(c.isArray()) {
                int n=Array.getLength(source); Object target=Array.newInstance(c.getComponentType(),n); seen.put(source,target);
                if(c.getComponentType().isPrimitive()) System.arraycopy(source,0,target,0,n);
                else for(int i=0;i<n;i++) Array.set(target,i,copy(Array.get(source,i)));
                return (T)target;
            }
            if(source instanceof Random) {
                ByteArrayOutputStream bytes=new ByteArrayOutputStream();
                new ObjectOutputStream(bytes).writeObject(source);
                Object target=new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray())).readObject();
                seen.put(source,target); return (T)target;
            }
            if(source instanceof Map) {
                Map target=source instanceof EnumMap ? new EnumMap((EnumMap)source) : new HashMap(); target.clear(); seen.put(source,target);
                for(Object item:((Map)source).entrySet()) { Map.Entry entry=(Map.Entry)item; target.put(copy(entry.getKey()),copy(entry.getValue())); }
                return (T)target;
            }
            if(source instanceof Collection) {
                Collection target=source instanceof Set ? new HashSet() : source instanceof Queue ? new LinkedList() : new ArrayList(); seen.put(source,target);
                for(Object value:(Collection)source) target.add(copy(value));
                return (T)target;
            }
            if(c.getName().startsWith("java.")) throw new IllegalStateException("Unsupported snapshot type: "+c.getName());
            Object target=U.allocateInstance(c); seen.put(source,target);
            for(Class<?> k=c;k!=Object.class;k=k.getSuperclass()) for(Field f:k.getDeclaredFields()) {
                if(Modifier.isStatic(f.getModifiers())) continue;
                f.setAccessible(true); Object value=f.get(source);
                f.set(target, f.getType().isPrimitive() || f.getName().equals("bfs_map_0") || f.getName().equals("bfs_map_1") ? value : copy(value));
            }
            return (T)target;
        } catch(Exception e) { throw new IllegalStateException("Cannot snapshot "+c.getName(),e); }
    }
}
