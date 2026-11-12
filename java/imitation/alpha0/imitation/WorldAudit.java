package alpha0.imitation;

import battlecode.world.GameWorld;
import battlecode.world.InternalRobot;
import java.io.*;
import java.lang.reflect.*;
import java.util.*;
import java.util.zip.DeflaterOutputStream;
import gnu.trove.TIntArrayList;
import gnu.trove.TIntObjectHashMap;

/** Privileged audit-only world graph. NEVER a student observation or model input. */
public final class WorldAudit {
    private static final Set<String> EXCLUDED=Set.of(
        "battlecode.world.GameWorld.controlProvider", "battlecode.world.GameWorld.matchMaker",
        "battlecode.world.GameWorld.profilerCollections", "battlecode.world.InternalRobot.controller",
        "battlecode.world.InternalRobot.gameWorld", "battlecode.world.InternalRobot.cachedRobotInfo");
    private final IdentityHashMap<Object,Integer> seen=new IdentityHashMap<>();
    private int nodes;
    public static Map<String,Object> capture(GameWorld world)throws IOException {
        WorldAudit audit=new WorldAudit();
        return Map.of("schema","engine-world-audit-v1","excluded_fields",EXCLUDED.stream().sorted().toList(),
            "world",audit.encode(world),"static_robot_rng",audit.encode(InternalRobot.rand));
    }
    private Object encode(Object value)throws IOException {
        if(++nodes>1000000)throw new IOException("audit node cap");
        if(value==null || value instanceof String || value instanceof Boolean || value instanceof Integer || value instanceof Long)return value;
        if(value instanceof Byte || value instanceof Short)return ((Number)value).intValue();
        if(value instanceof Character c)return (int)c;
        if(value instanceof Float f)return Map.of("float_hex",Float.toHexString(f));
        if(value instanceof Double d)return Map.of("double_hex",Double.toHexString(d));
        if(value instanceof Enum<?> e)return Map.of("enum",e.getDeclaringClass().getName(),"name",e.name());
        if(value instanceof Random r){var bytes=new ByteArrayOutputStream();try(var out=new ObjectOutputStream(bytes)){out.writeObject(r);}return Map.of("random_serialized",Base64.getEncoder().encodeToString(bytes.toByteArray()));}
        Class<?> type=value.getClass();
        if(type.isArray() && (type.getComponentType().isPrimitive() || type.getComponentType().isEnum())) {
            var bytes=new ByteArrayOutputStream();
            try(var compressed=new DeflaterOutputStream(bytes);var out=new ObjectOutputStream(compressed)){out.writeObject(value);}
            return Map.of("array_class",type.getName(),"length",Array.getLength(value),"java_serialized_zlib_base64",Base64.getEncoder().encodeToString(bytes.toByteArray()));
        }
        if(type.isArray()){List<Object> out=new ArrayList<>();for(int i=0;i<Array.getLength(value);i++)out.add(encode(Array.get(value,i)));return out;}
        if(value instanceof TIntArrayList a)return encode(a.toNativeArray());
        if(value instanceof TIntObjectHashMap<?> m){int[] keys=m.keys();List<Object> entries=new ArrayList<>();int[] sorted=keys.clone();Arrays.sort(sorted);for(int k:sorted)entries.add(Arrays.asList(k,encode(m.get(k))));return Map.of("trove_entries",entries,"iteration_keys",encode(keys));}
        if(value instanceof Map<?,?> m){
            List<Map.Entry<?,?>> entries=new ArrayList<>(m.entrySet());
            entries.sort(Comparator.comparing(e->key(e.getKey())));
            List<Object> out=new ArrayList<>();for(var e:entries)out.add(Arrays.asList(encode(e.getKey()),encode(e.getValue())));return Map.of("map_entries",out);
        }
        if(value instanceof Collection<?> c){List<?> ordered=new ArrayList<>(c);if(value instanceof Set<?>)ordered.sort(Comparator.comparing(WorldAudit::key));List<Object> out=new ArrayList<>();for(Object o:ordered)out.add(encode(o));return out;}
        if(!type.getName().startsWith("battlecode.world.")&&!type.getName().startsWith("battlecode.common."))throw new IOException("unreviewed audit type: "+type.getName());
        if(seen.containsKey(value))return Map.of("ref",seen.get(value));
        int id=seen.size();seen.put(value,id);Map<String,Object> fields=new TreeMap<>();
        List<Field> all=new ArrayList<>();for(Class<?> c=type;c!=Object.class;c=c.getSuperclass())all.addAll(Arrays.asList(c.getDeclaredFields()));
        all.sort(Comparator.comparing(f->f.getDeclaringClass().getName()+"."+f.getName()));
        for(Field f:all){if(Modifier.isStatic(f.getModifiers()))continue;String name=f.getDeclaringClass().getName()+"."+f.getName();if(EXCLUDED.contains(name))continue;
            try{f.setAccessible(true);fields.put(name,encode(f.get(value)));}catch(ReflectiveOperationException e){throw new IOException("audit field inaccessible: "+name,e);}}
        return Map.of("id",id,"class",type.getName(),"fields",fields);
    }
    private static String key(Object value){
        if(value instanceof Enum<?> e)return e.getDeclaringClass().getName()+":"+e.name();
        if(value instanceof String || value instanceof Integer || value instanceof Long)return value.getClass().getName()+":"+value;
        throw new IllegalArgumentException("unreviewed map/set key");
    }
    private WorldAudit(){}
}
