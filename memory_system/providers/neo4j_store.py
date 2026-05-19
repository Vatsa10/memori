"""Production Neo4j-backed graph store."""

from typing import Optional

from memory_system.core.memory_models import Entity, Relationship


class Neo4jGraphStore:
    """Entity-relationship graph store backed by Neo4j."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
        self._uri = uri
        self._user = user
        self._password = password
        self._driver = None

    def _get_driver(self):
        if self._driver is None:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self._uri, auth=(self._user, self._password)
            )
        return self._driver

    async def ensure_indexes(self):
        driver = self._get_driver()
        with driver.session() as session:
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.user_id)")

    async def add_entity(self, entity: Entity) -> None:
        driver = self._get_driver()
        with driver.session() as session:
            session.run(
                """
                MERGE (e:Entity {name: $name, user_id: $user_id})
                SET e.entity_type = $entity_type,
                    e += $properties
                """,
                name=entity.name,
                user_id=entity.user_id,
                entity_type=entity.entity_type,
                properties=entity.properties,
            )

    async def add_relationship(self, relationship: Relationship) -> None:
        driver = self._get_driver()
        with driver.session() as session:
            session.run(
                """
                MERGE (s:Entity {name: $source, user_id: $user_id})
                MERGE (t:Entity {name: $target, user_id: $user_id})
                MERGE (s)-[r:RELATES_TO {type: $rel_type}]->(t)
                SET r += $properties
                """,
                source=relationship.source_entity,
                target=relationship.target_entity,
                user_id=relationship.user_id,
                rel_type=relationship.relation_type,
                properties=relationship.properties,
            )

    async def search_entities(
        self, query: str, user_id: str, k: int = 5
    ) -> list[Entity]:
        driver = self._get_driver()
        with driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity {user_id: $user_id})
                WHERE toLower(e.name) CONTAINS toLower($query)
                   OR toLower(e.entity_type) CONTAINS toLower($query)
                RETURN e.name AS name, e.entity_type AS entity_type,
                       e.user_id AS user_id, properties(e) AS props
                LIMIT $k
                """,
                query=query,
                user_id=user_id,
                k=k,
            )
            entities = []
            for record in result:
                props = dict(record["props"])
                props.pop("name", None)
                props.pop("entity_type", None)
                props.pop("user_id", None)
                entities.append(Entity(
                    name=record["name"],
                    entity_type=record["entity_type"] or "unknown",
                    user_id=record["user_id"],
                    properties=props,
                ))
            return entities

    async def get_related(
        self,
        entity_name: str,
        user_id: str,
        relation_type: Optional[str] = None,
    ) -> list[Relationship]:
        driver = self._get_driver()
        with driver.session() as session:
            if relation_type:
                query = """
                    MATCH (s:Entity {user_id: $user_id})-[r:RELATES_TO {type: $rel_type}]-(t:Entity)
                    WHERE s.name = $name
                    RETURN s.name AS source, t.name AS target, r.type AS rel_type, properties(r) AS props
                """
                result = session.run(query, name=entity_name, user_id=user_id, rel_type=relation_type)
            else:
                query = """
                    MATCH (s:Entity {user_id: $user_id})-[r:RELATES_TO]-(t:Entity)
                    WHERE s.name = $name
                    RETURN s.name AS source, t.name AS target, r.type AS rel_type, properties(r) AS props
                """
                result = session.run(query, name=entity_name, user_id=user_id)

            relationships = []
            for record in result:
                props = dict(record["props"])
                props.pop("type", None)
                relationships.append(Relationship(
                    source_entity=record["source"],
                    target_entity=record["target"],
                    relation_type=record["rel_type"] or "related",
                    user_id=user_id,
                    properties=props,
                ))
            return relationships

    async def traverse(
        self,
        start_entity: str,
        user_id: str,
        max_hops: int = 2,
        relation_filter: Optional[list[str]] = None,
        max_results: int = 20,
    ) -> list[list[Relationship]]:
        """Variable-length path walk over outgoing relationships."""
        if max_hops < 1:
            return []

        driver = self._get_driver()
        # Cypher variable-length paths don't take a parameter for the bound,
        # so interpolate after clamping (1..50 sane range).
        hops = max(1, min(int(max_hops), 50))
        cypher = (
            f"MATCH path = (a:Entity {{name: $name, user_id: $user_id}})"
            f"-[rels:RELATES_TO*1..{hops}]->(b:Entity) "
            "WHERE ALL(rel IN rels WHERE rel.user_id IS NULL OR rel.user_id = $user_id) "
            + (
                "AND ALL(rel IN rels WHERE rel.type IN $relations) "
                if relation_filter
                else ""
            )
            + "RETURN [rel IN rels | {source: startNode(rel).name, "
            "target: endNode(rel).name, rel_type: rel.type, props: properties(rel)}] AS path "
            "LIMIT $max"
        )

        with driver.session() as session:
            result = session.run(
                cypher,
                name=start_entity,
                user_id=user_id,
                max=max_results,
                relations=relation_filter or [],
            )
            paths: list[list[Relationship]] = []
            for record in result:
                hops_data = record["path"]
                rels = []
                for hop in hops_data:
                    props = dict(hop.get("props", {}) or {})
                    props.pop("type", None)
                    rels.append(
                        Relationship(
                            source_entity=hop["source"],
                            target_entity=hop["target"],
                            relation_type=hop["rel_type"] or "related",
                            user_id=user_id,
                            properties=props,
                        )
                    )
                if rels:
                    paths.append(rels)
            return paths

    def close(self):
        if self._driver:
            self._driver.close()
